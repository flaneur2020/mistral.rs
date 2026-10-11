//! CPU kernels for the GGML **IQ4_XS** weight format (GGML type 23), x86-64.
//!
//! The published Qwen3.8-Flash-Next GGUFs store their routed experts and
//! several dense tensors in IQ4_XS, which `candle-core` cannot decode, so
//! `naiwa-rs` has to re-quantize them offline first. These kernels are the
//! first step towards running those files directly.
//!
//! `candle-core` cannot name the block type, so candle only carries a
//! `GgmlDType::IQ4XS` variant (its own storage hooks for it are stubs);
//! [`BlockIQ4XS`] lives here, implements candle's `GgmlType`, and
//! [`storage_for_device`] builds its `QStorage` on CPU or CUDA. On CUDA the
//! decode (`indexed_moe`) and matvec (`mmvq_gguf`) kernels dispatch to it the
//! same way they do for candle's own block types.
//!
//! ## CUDA
//!
//! [`BlockIQ4XS`]'s `vec_dot` runs on the CPU. The CUDA kernels all use the
//! same `vec_dot_iq4_xs_q8_1` ported from llama.cpp:
//! `kernels/indexed_moe/indexed_moe.cu` (indexed + fused decode),
//! `kernels/mmvq_gguf/mmvq_gguf.cu` (dense matvec), `kernels/moe_grouped/moe_grouped.cu`
//! (grouped prefill) and `kernels/mmq_gguf/mmq_instance_iq4_xs.cu` (dense prompt
//! and MoE pair MMQ; the type traits and tiles already shipped in `mmq_gguf.cuh`).
//!
//! # Layout
//!
//! Weights come in 256-element super-blocks of 136 bytes:
//!
//! ```text
//! offset  0: d          u16 (f16 bits)   super-block scale
//! offset  2: scales_h   u16              2 high bits of the 8 group scales
//! offset  4: scales_l   [u8; 4]          4 low bits of the 8 group scales
//! offset  8: qs         [u8; 128]        4-bit codebook indices
//! ```
//!
//! The super-block holds eight 32-element groups. Each group has a 6-bit signed
//! scale `ls = low4 | (high2 << 4)` read as `ls - 32`, so the effective scale is
//! `d * (ls - 32)` with `ls - 32` in `-32..=31`. A group's 32 values are 4-bit
//! indices into [`KVALUES_IQ4NL`], stored as the **16 low nibbles first, then the
//! 16 high nibbles** of the group's 16 bytes.
//!
//! Activations are quantized once per row into GGML's Q8_K block (292 bytes:
//! `f32 d`, `i8 qs[256]`, `i16 bsums[16]`); the dot product is then integer
//! arithmetic, as in `ggml_vec_dot_iq4_xs_q8_K`.

use candle_core::{
    bail,
    quantized::{
        k_quants::{BlockQ8K, GgmlType},
        GgmlDType, QStorage,
    },
    Device, Result,
};

/// Elements per IQ4_XS super-block.
pub const QK_K: usize = 256;
/// Bytes per packed IQ4_XS super-block.
pub const BLOCK_BYTES: usize = 136;
/// Bytes per packed Q8_K activation block.
pub const Q8K_BYTES: usize = 292;

/// The 16-entry non-linear codebook, shared with IQ4_NL.
///
/// It must stay sorted ascending: `best_index_int8` binary-searches it.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// `GROUP_MAX_EPS` from ggml: a group whose largest magnitude is below this is
/// stored with a zero scale.
const GROUP_MAX_EPS: f32 = 1e-15;

/// One packed IQ4_XS super-block, byte-compatible with ggml's `block_iq4_xs`.
///
/// The fields hold the packed bytes as-is. The kernels above view a `&[Self]`
/// as `&[u8]`, which is sound because the layout is `#[repr(C)]` and unpadded
/// (asserted below).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ4XS {
    /// Super-block scale, as f16 bits.
    pub d: u16,
    /// The two high bits of each of the eight 6-bit group scales.
    pub scales_h: u16,
    /// The four low bits of each of the eight 6-bit group scales.
    pub scales_l: [u8; 4],
    /// The 4-bit codebook indices, 16 bytes per 32-element group.
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockIQ4XS>() == BLOCK_BYTES);

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
fn rd_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// ggml's `nearest_int`: round to nearest, ties to even, through the 1.5*2^23
/// magic constant. Bit-identical to ggml for every finite input in range.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(
        fval.abs() <= 4_194_303.0,
        "nearest_int out of range: {fval}"
    );
    let val = fval + 12_582_912.0;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// Validate a `(weights, output)` pair and return the super-block count.
fn check_blocks(weight_len: usize, out_len: usize) -> Result<usize> {
    if !weight_len.is_multiple_of(BLOCK_BYTES) {
        bail!("IQ4_XS: weight length {weight_len} is not a multiple of {BLOCK_BYTES}");
    }
    let nb = weight_len / BLOCK_BYTES;
    if out_len != nb * QK_K {
        bail!(
            "IQ4_XS: output length {out_len} does not match {nb} super-blocks ({} elements)",
            nb * QK_K
        );
    }
    Ok(nb)
}

/// Decode `src.len() / 136` packed super-blocks into `dst`.
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

/// Portable scalar decoder, bit-identical to `dequantize_row_iq4_xs`.
pub fn dequantize_scalar(src: &[u8], dst: &mut [f32]) -> Result<()> {
    let nb = check_blocks(src.len(), dst.len())?;
    for b in 0..nb {
        dequantize_block(
            &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
            &mut dst[b * QK_K..(b + 1) * QK_K],
        );
    }
    Ok(())
}

fn dequantize_block(blk: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd_u16(blk, 0));
    let scales_h = rd_u16(blk, 2);
    let scales_l = &blk[4..8];
    let qs = &blk[8..BLOCK_BYTES];
    for ib in 0..8 {
        let ls = group_scale(scales_l, scales_h, ib);
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            let byte = qs[ib * 16 + j];
            out[ib * 32 + j] = dl * KVALUES_IQ4NL[(byte & 0xf) as usize] as f32;
            out[ib * 32 + 16 + j] = dl * KVALUES_IQ4NL[(byte >> 4) as usize] as f32;
        }
    }
}

/// The 6-bit signed scale of group `ib`, before the `- 32` bias.
#[inline]
fn group_scale(scales_l: &[u8], scales_h: u16, ib: usize) -> i32 {
    ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32 | (((scales_h >> (2 * ib)) & 3) as i32) << 4
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
        let out = dst.as_mut_ptr().add(b * QK_K);
        let d = f16_to_f32(rd_u16(blk, 0));
        let scales_h = rd_u16(blk, 2);
        let scales_l = &blk[4..8];
        let qs = blk.as_ptr().add(8);
        for ib in 0..8 {
            let dl = _mm256_set1_ps(d * (group_scale(scales_l, scales_h, ib) - 32) as f32);
            let byte = _mm_loadu_si128(qs.add(ib * 16) as *const __m128i);
            let lo = _mm_shuffle_epi8(values, _mm_and_si128(byte, m4b));
            let hi = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16::<4>(byte), m4b));
            for (half, base) in [(lo, 0usize), (hi, 16usize)] {
                let lo8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(half));
                let hi8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(half)));
                _mm256_storeu_ps(out.add(ib * 32 + base), _mm256_mul_ps(lo8, dl));
                _mm256_storeu_ps(out.add(ib * 32 + base + 8), _mm256_mul_ps(hi8, dl));
            }
        }
    }
    Ok(())
}

/// Quantize `x` into packed Q8_K blocks, bit-identical to
/// `quantize_row_q8_K_ref`.
///
/// The reference leaves `bsums` untouched for an all-zero block; this writes
/// zeros there instead. Nothing that consumes Q8_K reads `bsums`.
///
/// `candle-core`'s `BlockQ8K::from_float` does the same job for the wired-up
/// path; this exists so the kernel and its tests do not depend on private
/// candle fields. Kept as a standalone entry point for tests and benchmarks.
#[allow(dead_code)]
pub fn quantize_row_q8_k(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK_K) {
        bail!(
            "IQ4_XS: activation length {} is not a positive multiple of {QK_K}",
            x.len()
        );
    }
    let nb = x.len() / QK_K;
    if dst.len() != nb * Q8K_BYTES {
        bail!(
            "IQ4_XS: Q8_K output length {} does not match {nb} blocks ({})",
            dst.len(),
            nb * Q8K_BYTES
        );
    }
    for b in 0..nb {
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        let out = &mut dst[b * Q8K_BYTES..(b + 1) * Q8K_BYTES];
        let mut max = 0f32;
        let mut amax = 0f32;
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            out.fill(0);
            continue;
        }
        let iscale = -127.0 / max;
        out[..4].copy_from_slice(&(1.0f32 / iscale).to_le_bytes());
        for j in 0..QK_K {
            out[4 + j] = nearest_int(iscale * xb[j]).min(127) as i8 as u8;
        }
        for j in 0..16 {
            let sum: i32 = out[4 + j * 16..4 + j * 16 + 16]
                .iter()
                .map(|&q| q as i8 as i32)
                .sum();
            out[260 + j * 2..260 + j * 2 + 2].copy_from_slice(&(sum as i16).to_le_bytes());
        }
    }
    Ok(())
}

/// Dot product of packed IQ4_XS weights against packed Q8_K activations.
///
/// Uses AVX2 when the CPU has it.
pub fn vec_dot_q8_k(w: &[u8], q8: &[u8]) -> Result<f32> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { vec_dot_q8_k_avx2(w, q8) };
    }
    vec_dot_q8_k_scalar(w, q8)
}

fn check_dot(w: &[u8], q8: &[u8]) -> Result<usize> {
    if !w.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "IQ4_XS: weight length {} is not a multiple of {BLOCK_BYTES}",
            w.len()
        );
    }
    let nb = w.len() / BLOCK_BYTES;
    if q8.len() != nb * Q8K_BYTES {
        bail!(
            "IQ4_XS: Q8_K length {} does not match {nb} super-blocks ({})",
            q8.len(),
            nb * Q8K_BYTES
        );
    }
    Ok(nb)
}

/// Scalar dot, mirroring `ggml_vec_dot_iq4_xs_q8_K_generic`.
pub fn vec_dot_q8_k_scalar(w: &[u8], q8: &[u8]) -> Result<f32> {
    let nb = check_dot(w, q8)?;
    let mut sumf = 0f32;
    for b in 0..nb {
        let wb = &w[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let qb = &q8[b * Q8K_BYTES..(b + 1) * Q8K_BYTES];
        let d4d8 = f16_to_f32(rd_u16(wb, 0)) * rd_f32(qb, 0);
        let scales_l = &wb[4..8];
        let scales_h = rd_u16(wb, 2);
        let qs = &wb[8..BLOCK_BYTES];
        let q8s = &qb[4..260];
        let mut sh = scales_h;
        let mut off = 0usize;
        for ib in (0..8).step_by(2) {
            let ls1 = ((scales_l[ib / 2] & 0xf) as i32) | (((sh << 4) & 0x30) as i32);
            let ls2 = ((scales_l[ib / 2] >> 4) as i32) | (((sh << 2) & 0x30) as i32);
            sh >>= 4;
            sumf += d4d8 * (ls1 - 32) as f32 * dot_group(&q8s[off..], &qs[ib * 16..]) as f32;
            sumf +=
                d4d8 * (ls2 - 32) as f32 * dot_group(&q8s[off + 32..], &qs[ib * 16 + 16..]) as f32;
            off += 64;
        }
    }
    Ok(sumf)
}

/// One 32-element group: `sum q8[j] * kvalues[qs[j] nibble]`, signed int8 q8.
#[inline]
fn dot_group(q8: &[u8], qs: &[u8]) -> i32 {
    let mut sumi = 0i32;
    for j in 0..16 {
        sumi += (q8[j] as i8 as i32) * KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as i32;
        sumi += (q8[j + 16] as i8 as i32) * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as i32;
    }
    sumi
}

/// AVX2 dot, ported from `ggml_vec_dot_iq4_xs_q8_K`.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q8_k_avx2(w: &[u8], q8: &[u8]) -> Result<f32> {
    use std::arch::x86_64::*;
    let nb = check_dot(w, q8)?;
    let values = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mut accum = _mm256_setzero_ps();
    for b in 0..nb {
        let wb = &w[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let qb = &q8[b * Q8K_BYTES..(b + 1) * Q8K_BYTES];
        let mut qs = wb.as_ptr().add(8);
        let scales_l = wb.as_ptr().add(4);
        let scales_h = rd_u16(wb, 2);
        let mut sh = scales_h;
        let mut q8p = qb.as_ptr().add(4) as *const i8;
        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();
        for ib in (0..8).step_by(2) {
            let bits1 = _mm_loadu_si128(qs as *const __m128i);
            let bits2 = _mm_loadu_si128(qs.add(16) as *const __m128i);
            qs = qs.add(32);
            let q8b1 = _mm256_loadu_si256(q8p as *const __m256i);
            let q8b2 = _mm256_loadu_si256(q8p.add(32) as *const __m256i);
            q8p = q8p.add(64);
            let q4b1 = mm256_set_m128i(
                _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16::<4>(bits1), m4b)),
                _mm_shuffle_epi8(values, _mm_and_si128(bits1, m4b)),
            );
            let q4b2 = mm256_set_m128i(
                _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16::<4>(bits2), m4b)),
                _mm_shuffle_epi8(values, _mm_and_si128(bits2, m4b)),
            );
            let p1 = mul_add_epi8(q4b1, q8b1);
            let p2 = mul_add_epi8(q4b2, q8b2);
            let sl = *scales_l.add(ib / 2);
            let ls1 = (((sl & 0xf) as i32) | (((sh << 4) & 0x30) as i32)) - 32;
            let ls2 = (((sl >> 4) as i32) | (((sh << 2) & 0x30) as i32)) - 32;
            sh >>= 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(p1, _mm256_set1_epi16(ls1 as i16)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(p2, _mm256_set1_epi16(ls2 as i16)));
        }
        let d4d8 = f16_to_f32(rd_u16(wb, 0)) * rd_f32(qb, 0);
        accum = _mm256_add_ps(
            _mm256_mul_ps(
                _mm256_set1_ps(d4d8),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            ),
            accum,
        );
    }
    Ok(hsum_ps_256(accum))
}

/// `mul_add_epi8` from ggml: signed `a[i] * b[i]` with adjacent pairs summed
/// into `i16`.
///
/// # Safety
///
/// The CPU must support AVX2.
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
///
/// # Safety
///
/// The CPU must support AVX2.
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
///
/// # Safety
///
/// The CPU must support AVX2.
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

/// `y = W x` for `W` shaped `[rows, cols]` in packed IQ4_XS and a single `x`
/// row. The activation row is quantized to Q8_K once and reused for every row.
///
/// Standalone entry point for tests and benchmarks; the engine reaches the
/// kernels through `crate::gguf::cpu::routed_gemv` and `GgmlType` instead.
#[allow(dead_code)]
pub fn gemv(weights: &[u8], x: &[f32], out: &mut [f32]) -> Result<()> {
    let cols = x.len();
    if cols == 0 || !cols.is_multiple_of(QK_K) {
        bail!("IQ4_XS gemv: input length {cols} is not a positive multiple of {QK_K}");
    }
    let row_bytes = (cols / QK_K) * BLOCK_BYTES;
    if !weights.len().is_multiple_of(row_bytes) {
        bail!(
            "IQ4_XS gemv: weight length {} is not a multiple of the {row_bytes}-byte row",
            weights.len()
        );
    }
    let rows = weights.len() / row_bytes;
    if out.len() != rows {
        bail!(
            "IQ4_XS gemv: output length {} does not match {rows} rows",
            out.len()
        );
    }
    let mut q8 = vec![0u8; (cols / QK_K) * Q8K_BYTES];
    quantize_row_q8_k(x, &mut q8)?;
    for (row, o) in out.iter_mut().enumerate() {
        *o = vec_dot_q8_k(&weights[row * row_bytes..(row + 1) * row_bytes], &q8)?;
    }
    Ok(())
}

/// The smallest codebook index whose value is nearest to `x` (ggml's
/// `best_index_int8`). `val` must be sorted ascending.
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

/// Quantize `x` into packed IQ4_XS blocks, bit-identical to
/// `quantize_row_iq4_xs_ref` (ggml's `quantize_row_iq4_nl_impl` for the XS
/// super-block, `ntry = 7`, no importance weights).
///
/// This is both the convert-to-another-variant fallback and what the tests
/// compare bit-for-bit.
pub fn quantize_iq4_xs(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK_K) {
        bail!(
            "IQ4_XS: input length {} is not a positive multiple of {QK_K}",
            x.len()
        );
    }
    let nb = x.len() / QK_K;
    if dst.len() != nb * BLOCK_BYTES {
        bail!(
            "IQ4_XS: output length {} does not match {nb} super-blocks ({})",
            dst.len(),
            nb * BLOCK_BYTES
        );
    }
    for b in 0..nb {
        quantize_block(
            &x[b * QK_K..(b + 1) * QK_K],
            &mut dst[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
        );
    }
    Ok(())
}

fn quantize_block(xb: &[f32], out: &mut [u8]) {
    let mut l = [0u8; QK_K];
    let mut weight = [0f32; 32];
    let mut scales = [0f32; 8];

    let mut max_scale = 0f32;
    let mut amax_scale = 0f32;

    // First pass: an optimal signed scale per 32-element group.
    for ib in 0..8 {
        let group = &xb[ib * 32..ib * 32 + 32];
        let mut amax = 0f32;
        let mut max = 0f32;
        for (j, &v) in group.iter().enumerate() {
            weight[j] = v * v;
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        // ntry > 0, so the first guess is the negated one.
        let mut d = -max / KVALUES_IQ4NL[0] as f32;
        let id = 1.0 / d;
        let (mut sumqx, mut sumq2) = (0f32, 0f32);
        for (j, &v) in group.iter().enumerate() {
            let idx = best_index_int8(&KVALUES_IQ4NL, id * v);
            l[ib * 32 + j] = idx as u8;
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
            for (j, &v) in group.iter().enumerate() {
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
        scales[ib] = d;
        if d.abs() > amax_scale {
            amax_scale = d.abs();
            max_scale = d;
        }
    }

    // Second pass: 6-bit group scales relative to the super-block scale.
    let d = -max_scale / 32.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut scales_h = 0u16;
    let mut scales_l = [0u8; 4];
    for ib in 0..8 {
        let mut li = nearest_int(id * scales[ib]).clamp(-32, 31);
        let dl = d * li as f32;
        let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
        for j in 0..32 {
            l[ib * 32 + j] = best_index_int8(&KVALUES_IQ4NL, idl * xb[ib * 32 + j]) as u8;
        }
        li += 32;
        if ib % 2 == 0 {
            scales_l[ib / 2] = (li & 0xf) as u8;
        } else {
            scales_l[ib / 2] |= ((li & 0xf) as u8) << 4;
        }
        scales_h |= ((li >> 4) as u16) << (2 * ib);
    }

    // Pack the 4-bit indices: 16 low nibbles, then 16 high nibbles per group.
    out[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    out[2..4].copy_from_slice(&scales_h.to_le_bytes());
    out[4..8].copy_from_slice(&scales_l);
    for i in 0..8 {
        for j in 0..16 {
            out[8 + 16 * i + j] = l[32 * i + j] | (l[32 * i + 16 + j] << 4);
        }
    }
}

/// View a `#[repr(C)]`, padding-free block slice as its packed bytes.
///
/// # Safety
///
/// `T` must be `#[repr(C)]` with no padding. Only [`BlockIQ4XS`] and
/// `BlockQ8K` (both asserted padding-free) are ever passed.
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

impl GgmlType for BlockIQ4XS {
    const DTYPE: GgmlDType = GgmlDType::IQ4XS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        // SAFETY: `BlockIQ4XS` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes(xs) };
        dequantize(bytes, ys).expect("IQ4_XS to_float: block counts match");
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // SAFETY: `BlockIQ4XS` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes_mut(ys) };
        quantize_iq4_xs(xs, bytes).expect("IQ4_XS from_float: block counts match");
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK_K);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_k(w, q).expect("IQ4_XS vec_dot: block counts match")
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK_K);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_k_scalar(w, q).expect("IQ4_XS vec_dot: block counts match")
    }
}

/// Copy packed IQ4_XS bytes into `BlockIQ4XS` values.
fn blocks_from_bytes(data: &[u8]) -> Result<Vec<BlockIQ4XS>> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "IQ4_XS: {} bytes is not a multiple of the {BLOCK_BYTES}-byte super-block",
            data.len()
        );
    }
    let mut blocks = vec![
        BlockIQ4XS {
            d: 0,
            scales_h: 0,
            scales_l: [0; 4],
            qs: [0; 128],
        };
        data.len() / BLOCK_BYTES
    ];
    // SAFETY: `BlockIQ4XS` is repr(C) and unpadded (see the size assert).
    unsafe { as_bytes_mut(&mut blocks) }.copy_from_slice(data);
    Ok(blocks)
}

/// Build a CPU `QStorage` holding `data` as IQ4_XS super-blocks.
///
/// `candle-core`'s `GgmlDType::IQ4_XS` has no block type of its own (its
/// `from_data` is a stub), so the GGUF loader calls this instead of
/// `QStorage::from_data`.
pub fn cpu_storage(data: &[u8]) -> Result<QStorage> {
    Ok(QStorage::Cpu(Box::new(blocks_from_bytes(data)?)))
}

/// Build the storage for `data` on `device`.
///
/// CPU keeps the packed IQ4_XS blocks as-is ([`cpu_storage`]). CUDA uploads the
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
        Device::Cuda(_) => bail!("IQ4_XS on CUDA needs the `cuda` feature"),
        _ => bail!("IQ4_XS is only supported on the CPU and CUDA"),
    }
}

#[cfg(test)]
mod tests;
