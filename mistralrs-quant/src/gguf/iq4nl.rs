//! CPU kernels for the GGML **IQ4_NL** weight format (GGML type 20), x86-64.
//!
//! IQ4_NL is the 32-element sibling of IQ4_XS: the same 16-entry non-linear
//! codebook, but one f16 block scale instead of a super-block with eight 6-bit
//! group scales. It is the most common expert type in the published
//! Qwen3.8-Flash-Next GGUFs, so it gets the same treatment: the block type
//! lives here and implements candle's `GgmlType`, and [`storage_for_device`]
//! builds its `QStorage` on CPU and CUDA.
//!
//! # Layout
//!
//! ```text
//! offset  0: d    u16 (f16 bits)  block scale
//! offset  2: qs   [u8; 16]        4-bit codebook indices
//! ```
//!
//! Element `j` (0..16) uses the low nibble of `qs[j]`; element `j + 16` the high
//! nibble. A block is 32 elements / 18 bytes; the paired activation is Q8_0
//! (`f16 d`, `i8 qs[32]`).

use candle_core::{
    bail,
    quantized::{
        k_quants::{BlockQ8_0, GgmlType},
        GgmlDType, QStorage,
    },
    Device, Result,
};

/// Elements per IQ4_NL block.
pub const QK4_NL: usize = 32;
/// Bytes per packed IQ4_NL block.
pub const BLOCK_BYTES: usize = 18;
/// Bytes per packed Q8_0 activation block.
pub const Q8_0_BYTES: usize = 34;

/// The 16-entry non-linear codebook, shared with IQ4_XS.
///
/// It must stay sorted ascending: `best_index_int8` binary-searches it.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// `GROUP_MAX_EPS` from ggml: a block whose largest magnitude is below this is
/// stored with a zero scale.
const GROUP_MAX_EPS: f32 = 1e-15;

/// One packed IQ4_NL block, byte-compatible with ggml's `block_iq4_nl`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ4NL {
    /// Block scale, as f16 bits.
    pub d: u16,
    /// The 4-bit codebook indices, low nibbles then high nibbles.
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockIQ4NL>() == BLOCK_BYTES);

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

/// ggml's `best_index_int8`: nearest codebook entry. `val` must be ascending.
#[inline]
fn best_index_int8(val: &[i8; 16], x: f32) -> usize {
    if x <= val[0] as f32 {
        return 0;
    }
    if x >= val[15] as f32 {
        return 15;
    }
    let (mut ml, mut mu) = (0usize, 15usize);
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - (val[mu - 1] as f32) < (val[mu] as f32) - x {
        mu - 1
    } else {
        mu
    }
}

/// Validate a `(weights, output)` pair and return the block count.
fn check_blocks(weight_len: usize, out_len: usize) -> Result<usize> {
    if !weight_len.is_multiple_of(BLOCK_BYTES) {
        bail!("IQ4_NL: weight length {weight_len} is not a multiple of {BLOCK_BYTES}");
    }
    let nb = weight_len / BLOCK_BYTES;
    if out_len != nb * QK4_NL {
        bail!(
            "IQ4_NL: output length {out_len} does not match {nb} blocks ({} elements)",
            nb * QK4_NL
        );
    }
    Ok(nb)
}

/// Decode `len(src) / 18` packed blocks into `dst`.
///
/// Uses AVX2 when the CPU has it.
pub fn dequantize(src: &[u8], dst: &mut [f32]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { dequantize_avx2(src, dst) };
    }
    dequantize_scalar(src, dst)
}

/// Portable scalar decoder, bit-identical to `dequantize_row_iq4_nl`.
pub fn dequantize_scalar(src: &[u8], dst: &mut [f32]) -> Result<()> {
    let nb = check_blocks(src.len(), dst.len())?;
    for b in 0..nb {
        dequantize_block(
            &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
            &mut dst[b * QK4_NL..(b + 1) * QK4_NL],
        );
    }
    Ok(())
}

fn dequantize_block(blk: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd_u16(blk, 0));
    let qs = &blk[2..BLOCK_BYTES];
    for j in 0..16 {
        out[j] = d * KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
        out[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
    }
}

/// AVX2 decoder.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dequantize_avx2(src: &[u8], dst: &mut [f32]) -> Result<()> {
    use std::arch::x86_64::*;
    let nb = check_blocks(src.len(), dst.len())?;
    let values = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    for b in 0..nb {
        let blk = &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let out = dst.as_mut_ptr().add(b * QK4_NL);
        let d = _mm256_set1_ps(f16_to_f32(rd_u16(blk, 0)));
        let byte = _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i);
        let lo = _mm_shuffle_epi8(values, _mm_and_si128(byte, m4b));
        let hi = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16::<4>(byte), m4b));
        for (half, base) in [(lo, 0usize), (hi, 16usize)] {
            let lo8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(half));
            let hi8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(half)));
            _mm256_storeu_ps(out.add(base), _mm256_mul_ps(lo8, d));
            _mm256_storeu_ps(out.add(base + 8), _mm256_mul_ps(hi8, d));
        }
    }
    Ok(())
}

/// Quantize `x` into packed Q8_0 blocks, bit-identical to
/// `quantize_row_q8_0_ref`.
///
/// `candle-core`'s `BlockQ8_0::from_float` does the same job for the wired-up
/// path; this exists so the kernel and its tests do not depend on private
/// candle fields.
#[allow(dead_code)]
pub fn quantize_row_q8_0(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK4_NL) {
        bail!(
            "IQ4_NL: activation length {} is not a positive multiple of {QK4_NL}",
            x.len()
        );
    }
    let nb = x.len() / QK4_NL;
    if dst.len() != nb * Q8_0_BYTES {
        bail!(
            "IQ4_NL: Q8_0 output length {} does not match {nb} blocks ({})",
            dst.len(),
            nb * Q8_0_BYTES
        );
    }
    for b in 0..nb {
        let xb = &x[b * QK4_NL..(b + 1) * QK4_NL];
        let out = &mut dst[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES];
        let mut amax = 0f32;
        for &v in xb {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for j in 0..QK4_NL {
            out[2 + j] = (xb[j] * id).round() as i8 as u8;
        }
    }
    Ok(())
}

/// Dot product of packed IQ4_NL weights against packed Q8_0 activations.
///
/// Uses AVX2 when the CPU has it.
pub fn vec_dot_q8_0(w: &[u8], q8: &[u8]) -> Result<f32> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { vec_dot_q8_0_avx2(w, q8) };
    }
    vec_dot_q8_0_scalar(w, q8)
}

fn check_dot(w: &[u8], q8: &[u8]) -> Result<usize> {
    if !w.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "IQ4_NL: weight length {} is not a multiple of {BLOCK_BYTES}",
            w.len()
        );
    }
    let nb = w.len() / BLOCK_BYTES;
    if q8.len() != nb * Q8_0_BYTES {
        bail!(
            "IQ4_NL: Q8_0 length {} does not match {nb} blocks ({})",
            q8.len(),
            nb * Q8_0_BYTES
        );
    }
    Ok(nb)
}

/// Scalar dot, mirroring `ggml_vec_dot_iq4_nl_q8_0_generic`.
pub fn vec_dot_q8_0_scalar(w: &[u8], q8: &[u8]) -> Result<f32> {
    let nb = check_dot(w, q8)?;
    let mut sumf = 0f32;
    for b in 0..nb {
        sumf += sum_block(
            &w[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
            &q8[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES],
        );
    }
    Ok(sumf)
}

/// One block: `d_w * d_q8 * sum_i q8[i] * kvalues[nibble]`.
#[inline]
fn sum_block(wb: &[u8], qb: &[u8]) -> f32 {
    let d = f16_to_f32(rd_u16(wb, 0)) * f16_to_f32(rd_u16(qb, 0));
    let qs = &wb[2..BLOCK_BYTES];
    let q8 = &qb[2..Q8_0_BYTES];
    let mut s1 = 0i32;
    let mut s2 = 0i32;
    for j in 0..16 {
        s1 += KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as i32 * q8[j] as i8 as i32;
        s2 += KVALUES_IQ4NL[(qs[j] >> 4) as usize] as i32 * q8[j + 16] as i8 as i32;
    }
    d * (s1 + s2) as f32
}

/// AVX2 dot, ported from `ggml_vec_dot_iq4_nl_q8_0`.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q8_0_avx2(w: &[u8], q8: &[u8]) -> Result<f32> {
    use std::arch::x86_64::*;
    let nb = check_dot(w, q8)?;
    let values = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);
    let mut accum1 = _mm256_setzero_ps();
    let mut accum2 = _mm256_setzero_ps();
    let mut ib = 0;
    while ib + 1 < nb {
        let w0 = w.as_ptr().add(ib * BLOCK_BYTES);
        let w1 = w.as_ptr().add((ib + 1) * BLOCK_BYTES);
        let q0 = q8.as_ptr().add(ib * Q8_0_BYTES);
        let q1 = q8.as_ptr().add((ib + 1) * Q8_0_BYTES);
        let q4bits1 = _mm_loadu_si128(w0.add(2) as *const __m128i);
        let q4bits2 = _mm_loadu_si128(w1.add(2) as *const __m128i);
        let q8b1 = _mm256_loadu_si256(q0.add(2) as *const __m256i);
        let q8b2 = _mm256_loadu_si256(q1.add(2) as *const __m256i);
        let q4b1 = mm256_set_m128i(
            _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16::<4>(q4bits1), m4b)),
            _mm_shuffle_epi8(values, _mm_and_si128(q4bits1, m4b)),
        );
        let q4b2 = mm256_set_m128i(
            _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16::<4>(q4bits2), m4b)),
            _mm_shuffle_epi8(values, _mm_and_si128(q4bits2, m4b)),
        );
        let p1 = _mm256_madd_epi16(mul_add_epi8(q4b1, q8b1), mone);
        let p2 = _mm256_madd_epi16(mul_add_epi8(q4b2, q8b2), mone);
        let d1 = f16_to_f32(rd_u16(w, ib * BLOCK_BYTES)) * f16_to_f32(rd_u16(q8, ib * Q8_0_BYTES));
        let d2 = f16_to_f32(rd_u16(w, (ib + 1) * BLOCK_BYTES))
            * f16_to_f32(rd_u16(q8, (ib + 1) * Q8_0_BYTES));
        accum1 = _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(p1)),
            accum1,
        );
        accum2 = _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(d2), _mm256_cvtepi32_ps(p2)),
            accum2,
        );
        ib += 2;
    }
    let mut sumf = hsum_ps_256(_mm256_add_ps(accum1, accum2));
    if ib < nb {
        sumf += sum_block(
            &w[ib * BLOCK_BYTES..(ib + 1) * BLOCK_BYTES],
            &q8[ib * Q8_0_BYTES..(ib + 1) * Q8_0_BYTES],
        );
    }
    Ok(sumf)
}

/// `mul_add_epi8` from ggml: signed `a[i] * b[i]`, adjacent pairs summed into i16.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_add_epi8(
    a: std::arch::x86_64::__m256i,
    b: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let ax = _mm256_sign_epi8(a, a);
    let sy = _mm256_sign_epi8(b, a);
    _mm256_maddubs_epi16(ax, sy)
}

/// `MM256_SET_M128I(hi, lo)` from ggml.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mm256_set_m128i(
    hi: std::arch::x86_64::__m128i,
    lo: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m256i {
    std::arch::x86_64::_mm256_inserti128_si256::<1>(
        std::arch::x86_64::_mm256_castsi128_si256(lo),
        hi,
    )
}

/// Horizontal sum of eight `f32`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_ps_256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// `y = W x` for `W` shaped `[rows, cols]` in packed IQ4_NL and a single `x`
/// row. The activation row is quantized to Q8_0 once and reused for every row.
///
/// Standalone entry point for tests and benchmarks; the engine reaches the
/// kernels through `crate::gguf::cpu::routed_gemv` and `GgmlType` instead.
#[allow(dead_code)]
pub fn gemv(weights: &[u8], x: &[f32], out: &mut [f32]) -> Result<()> {
    let cols = x.len();
    if cols == 0 || !cols.is_multiple_of(QK4_NL) {
        bail!("IQ4_NL gemv: input length {cols} is not a positive multiple of {QK4_NL}");
    }
    let row_bytes = (cols / QK4_NL) * BLOCK_BYTES;
    if !weights.len().is_multiple_of(row_bytes) {
        bail!(
            "IQ4_NL gemv: weight length {} is not a multiple of the {row_bytes}-byte row",
            weights.len()
        );
    }
    let rows = weights.len() / row_bytes;
    if out.len() != rows {
        bail!(
            "IQ4_NL gemv: output length {} does not match {rows} rows",
            out.len()
        );
    }
    let mut q8 = vec![0u8; (cols / QK4_NL) * Q8_0_BYTES];
    quantize_row_q8_0(x, &mut q8)?;
    for (row, o) in out.iter_mut().enumerate() {
        *o = vec_dot_q8_0(&weights[row * row_bytes..(row + 1) * row_bytes], &q8)?;
    }
    Ok(())
}

/// Quantize `x` into packed IQ4_NL blocks, bit-identical to
/// `quantize_row_iq4_nl_ref` (ggml's `quantize_row_iq4_nl_impl` for a single
/// 32-element block, `ntry = 7`, no importance weights). Also the
/// convert-to-another-variant fallback.
pub fn quantize_iq4_nl(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK4_NL) {
        bail!(
            "IQ4_NL: input length {} is not a positive multiple of {QK4_NL}",
            x.len()
        );
    }
    let nb = x.len() / QK4_NL;
    if dst.len() != nb * BLOCK_BYTES {
        bail!(
            "IQ4_NL: output length {} does not match {nb} blocks ({})",
            dst.len(),
            nb * BLOCK_BYTES
        );
    }
    for b in 0..nb {
        quantize_block(
            &x[b * QK4_NL..(b + 1) * QK4_NL],
            &mut dst[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
        );
    }
    Ok(())
}

fn quantize_block(xb: &[f32], out: &mut [u8]) {
    let mut l = [0u8; QK4_NL];
    let mut weight = [0f32; QK4_NL];

    let mut amax = 0f32;
    let mut max = 0f32;
    for (j, &v) in xb.iter().enumerate() {
        weight[j] = v * v;
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    let mut scale = 0f32;
    if amax >= GROUP_MAX_EPS {
        // ntry > 0, so the first guess is the negated one.
        let mut d = -max / KVALUES_IQ4NL[0] as f32;
        let id = 1.0 / d;
        let (mut sumqx, mut sumq2) = (0f32, 0f32);
        for (j, &v) in xb.iter().enumerate() {
            let idx = best_index_int8(&KVALUES_IQ4NL, id * v);
            let q = KVALUES_IQ4NL[idx] as f32;
            sumqx += weight[j] * q * v;
            sumq2 += weight[j] * q * q;
        }
        d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let mut best = d * sumqx;
        for itry in -7i32..=7 {
            let id = (itry as f32 + KVALUES_IQ4NL[0] as f32) / max;
            sumqx = 0.0;
            sumq2 = 0.0;
            for (j, &v) in xb.iter().enumerate() {
                let idx = best_index_int8(&KVALUES_IQ4NL, id * v);
                let q = KVALUES_IQ4NL[idx] as f32;
                sumqx += weight[j] * q * v;
                sumq2 += weight[j] * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scale = d;
    }

    out[..2].copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
    let id = if scale != 0.0 { 1.0 / scale } else { 0.0 };
    for (j, &v) in xb.iter().enumerate() {
        l[j] = best_index_int8(&KVALUES_IQ4NL, id * v) as u8;
    }
    for j in 0..16 {
        out[2 + j] = l[j] | (l[j + 16] << 4);
    }
}

/// View a `#[repr(C)]`, padding-free block slice as its packed bytes.
///
/// # Safety
///
/// `T` must be `#[repr(C)]` with no padding. Only [`BlockIQ4NL`] and
/// `BlockQ8_0` (both asserted padding-free by candle) are ever passed.
unsafe fn as_bytes<T>(xs: &[T]) -> &[u8] {
    std::slice::from_raw_parts(xs.as_ptr().cast::<u8>(), std::mem::size_of_val(xs))
}

/// Mutable version of [`as_bytes`].
///
/// # Safety
///
/// See [`as_bytes`].
unsafe fn as_bytes_mut<T>(xs: &mut [T]) -> &mut [u8] {
    std::slice::from_raw_parts_mut(xs.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(xs))
}

impl GgmlType for BlockIQ4NL {
    const DTYPE: GgmlDType = GgmlDType::IQ4NL;
    const BLCK_SIZE: usize = QK4_NL;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        // SAFETY: `BlockIQ4NL` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes(xs) };
        dequantize(bytes, ys).expect("IQ4_NL to_float: block counts match");
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // SAFETY: `BlockIQ4NL` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes_mut(ys) };
        quantize_iq4_nl(xs, bytes).expect("IQ4_NL from_float: block counts match");
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK4_NL);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_0(w, q).expect("IQ4_NL vec_dot: block counts match")
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK4_NL);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_0_scalar(w, q).expect("IQ4_NL vec_dot: block counts match")
    }
}

/// Copy packed IQ4_NL bytes into `BlockIQ4NL` values.
fn blocks_from_bytes(data: &[u8]) -> Result<Vec<BlockIQ4NL>> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "IQ4_NL: {} bytes is not a multiple of the {BLOCK_BYTES}-byte block",
            data.len()
        );
    }
    let mut blocks = vec![BlockIQ4NL { d: 0, qs: [0; 16] }; data.len() / BLOCK_BYTES];
    // SAFETY: `BlockIQ4NL` is repr(C) and unpadded (see the size assert).
    unsafe { as_bytes_mut(&mut blocks) }.copy_from_slice(data);
    Ok(blocks)
}

/// Build a CPU `QStorage` holding `data` as IQ4_NL blocks.
///
/// `candle-core`'s `GgmlDType::IQ4_NL` has no block type of its own (its
/// `from_data` is a stub), so the GGUF loader calls this instead of
/// `QStorage::from_data`.
pub fn cpu_storage(data: &[u8]) -> Result<QStorage> {
    Ok(QStorage::Cpu(Box::new(blocks_from_bytes(data)?)))
}

/// Build the storage for `data` on `device`.
///
/// CPU keeps the packed IQ4_NL blocks as-is ([`cpu_storage`]). CUDA uploads the
/// same bytes through candle's `cuda::load_quantized`, which only needs the
/// block's `DTYPE` and size; the kernels are the fork's.
pub fn storage_for_device(data: &[u8], device: &Device) -> Result<QStorage> {
    match device {
        Device::Cpu => cpu_storage(data),
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            let blocks = blocks_from_bytes(data)?;
            candle_core::quantized::cuda::load_quantized(dev, &blocks)
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => bail!("IQ4_NL on CUDA needs the `cuda` feature"),
        _ => bail!("IQ4_NL is only supported on the CPU and CUDA"),
    }
}

#[cfg(test)]
mod tests;
