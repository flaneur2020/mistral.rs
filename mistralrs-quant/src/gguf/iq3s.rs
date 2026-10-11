//! CPU kernels for the GGML **IQ3_S** weight format (GGML type 21), x86-64.
//!
//! A 256-element super-block: one f16 scale, 64 grid-index bytes, eight high-bit
//! bytes, 32 sign bytes and four 4-bit scale-pair bytes. The magnitudes come from
//! `IQ3S_GRID` (four values per 9-bit grid index, the top bit from `qh`). The
//! paired activation is Q8_K.
//!
//! The **read path** (dequantize + vec-dot) is ported from ggml and checked
//! against it, with an AVX2 fast path for the vec-dot. The quantizer is a plain
//! grid search, not ggml's map/neighbour search, so `from_float` produces valid
//! blocks that do not match ggml byte for byte; nothing in the load path calls it.

use candle_core::{
    bail,
    quantized::{
        k_quants::{BlockQ8K, GgmlType},
        GgmlDType, QStorage,
    },
    Device, Result,
};

mod tables;

use tables::{IQ3S_GRID, KMASK_IQ2XS};

/// Elements per IQ3_S super-block.
pub const QK_K: usize = 256;
/// Bytes per packed IQ3_S super-block (`d` + 64 + 8 + 32 + 4).
pub const BLOCK_BYTES: usize = 110;
/// Bytes per packed Q8_K activation block.
pub const Q8K_BYTES: usize = 292;

/// One packed IQ3_S super-block, byte-compatible with ggml's `block_iq3_s`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ3S {
    /// Super-block scale, as f16 bits.
    pub d: u16,
    /// 64 grid-index bytes (two per 8-element group).
    pub qs: [u8; 64],
    /// The high bit of each of the eight grid indices per 32-element group.
    pub qh: [u8; 8],
    /// One sign byte per 8-element group.
    pub signs: [u8; 32],
    /// Four scale bytes; low nibble for the first 32 of a 64-element pair.
    pub scales: [u8; 4],
}

const _: () = assert!(std::mem::size_of::<BlockIQ3S>() == BLOCK_BYTES);

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

/// ggml's `nearest_int` (round to nearest, ties to even).
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// Split a packed super-block into its four 16-byte halves of grid indices,
/// eight high-bit bytes, sign bytes and scales.
#[inline]
fn parts(blk: &[u8]) -> (&[u8], &[u8], &[u8], &[u8]) {
    (&blk[2..66], &blk[66..74], &blk[74..106], &blk[106..110])
}

fn check_blocks(weight_len: usize, out_len: usize) -> Result<usize> {
    if !weight_len.is_multiple_of(BLOCK_BYTES) {
        bail!("IQ3_S: weight length {weight_len} is not a multiple of {BLOCK_BYTES}");
    }
    let nb = weight_len / BLOCK_BYTES;
    if out_len != nb * QK_K {
        bail!(
            "IQ3_S: output length {out_len} does not match {nb} super-blocks ({} elements)",
            nb * QK_K
        );
    }
    Ok(nb)
}

/// Decode `len(src) / 110` packed super-blocks into `dst`, bit-identical to
/// `dequantize_row_iq3_s`.
pub fn dequantize(src: &[u8], dst: &mut [f32]) -> Result<()> {
    let nb = check_blocks(src.len(), dst.len())?;
    for b in 0..nb {
        let blk = &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let out = &mut dst[b * QK_K..(b + 1) * QK_K];
        let d = f16_to_f32(rd_u16(blk, 0));
        let (qs, qh, signs, sc) = parts(blk);
        for k in 0..QK_K / 64 {
            let db1 = d * (1.0 + 2.0 * (sc[k] & 0xf) as f32);
            let db2 = d * (1.0 + 2.0 * (sc[k] >> 4) as f32);
            for (half, db) in [(0usize, db1), (1usize, db2)] {
                let qh_h = qh[2 * k + half] as usize;
                let qs_h = &qs[8 * (2 * k + half)..8 * (2 * k + half) + 8];
                let signs_h = &signs[4 * (2 * k + half)..4 * (2 * k + half) + 4];
                for l in 0..4 {
                    let gi1 = qs_h[2 * l] as usize | ((qh_h << (8 - 2 * l)) & 0x100);
                    let gi2 = qs_h[2 * l + 1] as usize | ((qh_h << (7 - 2 * l)) & 0x100);
                    let g1 = IQ3S_GRID[gi1].to_le_bytes();
                    let g2 = IQ3S_GRID[gi2].to_le_bytes();
                    let s = signs_h[l];
                    let base = 64 * k + 32 * half + 8 * l;
                    for j in 0..4 {
                        out[base + j] =
                            db * g1[j] as f32 * if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        out[base + j + 4] = db
                            * g2[j] as f32
                            * if s & KMASK_IQ2XS[j + 4] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                    }
                }
            }
        }
    }
    Ok(())
}

/// Quantize `x` into packed Q8_K blocks, bit-identical to
/// `quantize_row_q8_K_ref` (bsums zeroed for an all-zero block).
#[allow(dead_code)]
pub fn quantize_row_q8_k(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK_K) {
        bail!(
            "IQ3_S: activation length {} is not a positive multiple of {QK_K}",
            x.len()
        );
    }
    let nb = x.len() / QK_K;
    if dst.len() != nb * Q8K_BYTES {
        bail!(
            "IQ3_S: Q8_K output length {} does not match {nb} blocks ({})",
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

/// Dot product of packed IQ3_S weights against packed Q8_K activations.
///
/// Uses AVX2 when the CPU has it.
pub fn vec_dot_q8_k(w: &[u8], q8: &[u8]) -> Result<f32> {
    check_dot(w, q8)?;
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
            "IQ3_S: weight length {} is not a multiple of {BLOCK_BYTES}",
            w.len()
        );
    }
    let nb = w.len() / BLOCK_BYTES;
    if q8.len() != nb * Q8K_BYTES {
        bail!(
            "IQ3_S: Q8_K length {} does not match {nb} super-blocks ({})",
            q8.len(),
            nb * Q8K_BYTES
        );
    }
    Ok(nb)
}

/// Scalar dot, mirroring `ggml_vec_dot_iq3_s_q8_K_generic`.
pub fn vec_dot_q8_k_scalar(w: &[u8], q8: &[u8]) -> Result<f32> {
    let nb = check_dot(w, q8)?;
    let mut sumf = 0f32;
    for b in 0..nb {
        let wb = &w[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let qb = &q8[b * Q8K_BYTES..(b + 1) * Q8K_BYTES];
        let d = f16_to_f32(rd_u16(wb, 0)) * rd_f32(qb, 0);
        let (qs, qh, signs, sc) = parts(wb);
        let q8s = &qb[4..260];
        let mut bsum = 0i32;
        for k in 0..QK_K / 64 {
            let ls1 = 1 + 2 * (sc[k] & 0xf) as i32;
            let ls2 = 1 + 2 * (sc[k] >> 4) as i32;
            for (half, ls) in [(0usize, ls1), (1usize, ls2)] {
                let qh_h = qh[2 * k + half] as usize;
                let qs_h = &qs[8 * (2 * k + half)..8 * (2 * k + half) + 8];
                let signs_h = &signs[4 * (2 * k + half)..4 * (2 * k + half) + 4];
                let mut sumi = 0i32;
                for l in 0..4 {
                    let gi1 = qs_h[2 * l] as usize | ((qh_h << (8 - 2 * l)) & 0x100);
                    let gi2 = qs_h[2 * l + 1] as usize | ((qh_h << (7 - 2 * l)) & 0x100);
                    let g1 = IQ3S_GRID[gi1].to_le_bytes();
                    let g2 = IQ3S_GRID[gi2].to_le_bytes();
                    let s = signs_h[l];
                    let base = 64 * k + 32 * half + 8 * l;
                    for j in 0..4 {
                        let s1 = if s & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                        let s2 = if s & KMASK_IQ2XS[j + 4] != 0 { -1 } else { 1 };
                        sumi += q8s[base + j] as i8 as i32 * g1[j] as i32 * s1;
                        sumi += q8s[base + j + 4] as i8 as i32 * g2[j] as i32 * s2;
                    }
                }
                bsum += ls * sumi;
            }
        }
        sumf += d * bsum as f32;
    }
    Ok(sumf)
}

/// Sum the eight `f32` lanes of `x` (ggml's `hsum_float_8`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_float_8(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(x);
    let hi = _mm256_extractf128_ps(x, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55)))
}

/// `k_mask1` from ggml's AVX2 IQ3_S dot: pick byte 0/1/2/3 of each 32-bit lane.
#[cfg(target_arch = "x86_64")]
const IQ3S_MASK1: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
];

/// `k_mask2`: one bit per sign, repeated four times.
#[cfg(target_arch = "x86_64")]
const IQ3S_MASK2: [u8; 32] = [
    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4,
    8, 16, 32, 64, 128,
];

/// AVX2 dot, mirroring the `__AVX2__` branch of `ggml_vec_dot_iq3_s_q8_K`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q8_k_avx2(w: &[u8], q8: &[u8]) -> Result<f32> {
    use std::arch::x86_64::*;
    let nb = w.len() / BLOCK_BYTES;
    let mask1 = _mm256_loadu_si256(IQ3S_MASK1.as_ptr() as *const __m256i);
    let mask2 = _mm256_loadu_si256(IQ3S_MASK2.as_ptr() as *const __m256i);
    let mut accumf = _mm256_setzero_ps();
    for i in 0..nb {
        let wb = w.as_ptr().add(i * BLOCK_BYTES);
        let qb = q8.as_ptr().add(i * Q8K_BYTES);
        let d = f16_to_f32(rd_u16(w, i * BLOCK_BYTES)) * rd_f32(q8, i * Q8K_BYTES);
        let mut qs = wb.add(2);
        let qh = wb.add(66);
        let mut signs = wb.add(74) as *const u16;
        let scales = wb.add(106);
        let mut q8p = qb.add(4);
        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();
        for ib32 in (0..8).step_by(2) {
            let q8_1 = _mm256_loadu_si256(q8p as *const __m256i);
            let q8_2 = _mm256_loadu_si256(q8p.add(32) as *const __m256i);
            q8p = q8p.add(64);
            let mut idx = [0usize; 16];
            for (j, slot) in idx.iter_mut().enumerate() {
                *slot = (*qs.add(j) as usize)
                    | (((*qh.add(ib32 + j / 8) as usize) << (8 - j % 8)) & 0x100);
            }
            qs = qs.add(16);
            let g = |j: usize| *IQ3S_GRID.get_unchecked(idx[j]) as i32;
            let q2_1 = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
            let q2_2 = _mm256_set_epi32(g(15), g(14), g(13), g(12), g(11), g(10), g(9), g(8));
            let s0 = signs.read_unaligned() as u32;
            let s1 = signs.add(1).read_unaligned() as u32;
            let s2 = signs.add(2).read_unaligned() as u32;
            let s3 = signs.add(3).read_unaligned() as u32;
            signs = signs.add(4);
            let sign_mask = |packed: u32| {
                let aux = _mm256_set1_epi32(packed as i32);
                _mm256_cmpeq_epi8(
                    _mm256_and_si256(_mm256_shuffle_epi8(aux, mask1), mask2),
                    mask2,
                )
            };
            let sg1 = sign_mask(s0 | (s1 << 16));
            let sg2 = sign_mask(s2 | (s3 << 16));
            let q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(sg1, q8_1), sg1);
            let q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(sg2, q8_2), sg2);
            let sc = *scales.add(ib32 / 2) as i16;
            let p1 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(q2_1, q8s_1),
                _mm256_set1_epi16(2 * (sc & 0xf) + 1),
            );
            let p2 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(q2_2, q8s_2),
                _mm256_set1_epi16(2 * (sc >> 4) + 1),
            );
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_add_ps(
            _mm256_mul_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            ),
            accumf,
        );
    }
    Ok(hsum_float_8(accumf))
}

/// Quantize `x` into packed IQ3_S blocks.
///
/// **Not ggml's quantizer** (see the module docs): a plain search over
/// `IQ3S_GRID` with a per-32-element group-scale search. Valid blocks, not
/// byte-identical.
pub fn quantize_iq3_s(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK_K) {
        bail!(
            "IQ3_S: input length {} is not a positive multiple of {QK_K}",
            x.len()
        );
    }
    let nb = x.len() / QK_K;
    if dst.len() != nb * BLOCK_BYTES {
        bail!(
            "IQ3_S: output length {} does not match {nb} super-blocks ({})",
            dst.len(),
            nb * BLOCK_BYTES
        );
    }
    for b in 0..nb {
        let blk = &mut dst[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        // Largest decoded magnitude is `d * 31 * 15`.
        let d_model = if amax > 0.0 {
            amax / (31.0 * 15.0)
        } else {
            0.0
        };
        let d_f16 = half::f16::from_f32(d_model);
        let d_used = d_f16.to_f32();
        let (d_part, rest) = blk.split_at_mut(2);
        d_part.copy_from_slice(&d_f16.to_bits().to_le_bytes());
        let (qs_part, rest) = rest.split_at_mut(64);
        let (qh, rest) = rest.split_at_mut(8);
        let (signs, sc) = rest.split_at_mut(32);

        for g in 0..QK_K / 32 {
            let group = &xb[g * 32..g * 32 + 32];
            let mut best_err = f32::INFINITY;
            let mut best = (0u8, [0usize; 8], [0u8; 4]);
            for cand in 0..16u32 {
                let db = d_used * (1 + 2 * cand) as f32;
                if db == 0.0 {
                    continue;
                }
                let mut err = 0f32;
                let mut gg = [0usize; 8];
                let mut ss = [0u8; 4];
                for l in 0..4 {
                    let sub = &group[l * 8..l * 8 + 8];
                    let mut mask = 0u8;
                    for (j, &v) in sub.iter().enumerate() {
                        if v < 0.0 {
                            mask |= 1 << j;
                        }
                    }
                    let mut t1 = [0f32; 4];
                    let mut t2 = [0f32; 4];
                    for j in 0..4 {
                        let s1 = if mask & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        let s2 = if mask & KMASK_IQ2XS[j + 4] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        t1[j] = s1 * sub[j] / db;
                        t2[j] = s2 * sub[j + 4] / db;
                    }
                    let (g1, e1) = best_grid4(&t1);
                    let (g2, e2) = best_grid4(&t2);
                    err += e1 + e2;
                    gg[2 * l] = g1;
                    gg[2 * l + 1] = g2;
                    ss[l] = mask;
                }
                if err < best_err {
                    best_err = err;
                    best = (cand as u8, gg, ss);
                }
            }
            sc[g / 2] |= best.0 << (4 * (g % 2));
            qh[g] = 0;
            for l in 0..4 {
                grid_store(qs_part, g, 2 * l, best.1[2 * l], &mut qh[g]);
                grid_store(qs_part, g, 2 * l + 1, best.1[2 * l + 1], &mut qh[g]);
                signs[4 * g + l] = best.2[l];
            }
        }
    }
    Ok(())
}

/// Write one 9-bit grid index into `qs`/`qh` at group `g`, slot `l`.
#[inline]
fn grid_store(qs: &mut [u8], g: usize, l: usize, index: usize, qh: &mut u8) {
    qs[8 * g + l] = (index & 0xff) as u8;
    *qh |= (((index >> 8) & 1) as u8) << l;
}

/// Nearest `IQ3S_GRID` entry to four target values (signed; a grid byte is in
/// `1..=15`, so a negative target is best served by the smallest entry).
fn best_grid4(target: &[f32; 4]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for (g, &entry) in IQ3S_GRID.iter().enumerate() {
        let bytes = entry.to_le_bytes();
        let mut err = 0f32;
        for j in 0..4 {
            let d = bytes[j] as f32 - target[j];
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best = g;
        }
    }
    (best, best_err)
}

/// View a `#[repr(C)]`, padding-free block slice as its packed bytes.
///
/// # Safety
///
/// `T` must be `#[repr(C)]` with no padding. Only [`BlockIQ3S`] and [`BlockQ8K`]
/// (asserted padding-free by candle) are ever passed.
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

impl GgmlType for BlockIQ3S {
    const DTYPE: GgmlDType = GgmlDType::IQ3_S;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        // SAFETY: `BlockIQ3S` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes(xs) };
        dequantize(bytes, ys).expect("IQ3_S to_float: block counts match");
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // SAFETY: `BlockIQ3S` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes_mut(ys) };
        quantize_iq3_s(xs, bytes).expect("IQ3_S from_float: block counts match");
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK_K);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_k(w, q).expect("IQ3_S vec_dot: block counts match")
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK_K);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_k(w, q).expect("IQ3_S vec_dot: block counts match")
    }
}

/// Copy packed IQ3_S bytes into `BlockIQ3S` values.
fn blocks_from_bytes(data: &[u8]) -> Result<Vec<BlockIQ3S>> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "IQ3_S: {} bytes is not a multiple of the {BLOCK_BYTES}-byte super-block",
            data.len()
        );
    }
    let mut blocks = vec![
        BlockIQ3S {
            d: 0,
            qs: [0; 64],
            qh: [0; 8],
            signs: [0; 32],
            scales: [0; 4],
        };
        data.len() / BLOCK_BYTES
    ];
    // SAFETY: `BlockIQ3S` is repr(C) and unpadded (see the size assert).
    unsafe { as_bytes_mut(&mut blocks) }.copy_from_slice(data);
    Ok(blocks)
}

/// Build a CPU `QStorage` holding `data` as IQ3_S super-blocks.
pub fn cpu_storage(data: &[u8]) -> Result<QStorage> {
    Ok(QStorage::Cpu(Box::new(blocks_from_bytes(data)?)))
}

/// Build the storage for `data` on `device`.
pub fn storage_for_device(data: &[u8], device: &Device) -> Result<QStorage> {
    match device {
        Device::Cpu => cpu_storage(data),
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            let blocks = blocks_from_bytes(data)?;
            candle_core::quantized::cuda::load_quantized(dev, &blocks)
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => bail!("IQ3_S on CUDA needs the `cuda` feature"),
        _ => bail!("IQ3_S is only supported on the CPU and CUDA"),
    }
}

/// `y = W x` for `W` shaped `[rows, cols]` in packed IQ3_S and a single `x` row,
/// the activation quantized to Q8_K once.
#[allow(dead_code)]
pub fn gemv(weights: &[u8], x: &[f32], out: &mut [f32]) -> Result<()> {
    let cols = x.len();
    if cols == 0 || !cols.is_multiple_of(QK_K) {
        bail!("IQ3_S gemv: input length {cols} is not a positive multiple of {QK_K}");
    }
    let row_bytes = (cols / QK_K) * BLOCK_BYTES;
    if !weights.len().is_multiple_of(row_bytes) {
        bail!(
            "IQ3_S gemv: weight length {} is not a multiple of the {row_bytes}-byte row",
            weights.len()
        );
    }
    let rows = weights.len() / row_bytes;
    if out.len() != rows {
        bail!(
            "IQ3_S gemv: output length {} does not match {rows} rows",
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

#[cfg(test)]
mod tests;
