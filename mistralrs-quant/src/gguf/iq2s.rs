//! CPU kernels for the GGML **IQ2_S** weight format (GGML type 22), x86-64.
//!
//! A 256-element super-block: one f16 scale, 32 grid-index bytes plus 32 sign
//! bytes, eight high-bit bytes, and eight scale bytes. The magnitudes come from
//! `IQ2S_GRID` (eight values per 10-bit grid index, the top two bits from `qh`).
//! The paired activation is Q8_K.
//!
//! The **read path** (dequantize + vec-dot) is ported from ggml and checked
//! against it. The quantizer is a plain grid search, not ggml's
//! map/neighbour search, so `from_float` produces valid blocks that do not match
//! ggml byte for byte; nothing in the load path calls it.

use candle_core::{
    bail,
    quantized::{
        k_quants::{BlockQ8K, GgmlType},
        GgmlDType, QStorage,
    },
    Device, Result,
};

mod tables;

use tables::{IQ2S_GRID, KMASK_IQ2XS};

/// Elements per IQ2_S super-block.
pub const QK_K: usize = 256;
/// Bytes per packed IQ2_S super-block (`d` + 64 + 8 + 8).
pub const BLOCK_BYTES: usize = 82;
/// Bytes per packed Q8_K activation block.
pub const Q8K_BYTES: usize = 292;

/// One packed IQ2_S super-block, byte-compatible with ggml's `block_iq2_s`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ2S {
    /// Super-block scale, as f16 bits.
    pub d: u16,
    /// 32 grid-index bytes then 32 sign bytes.
    pub qs: [u8; 64],
    /// The two high bits of each of the four grid indices per group.
    pub qh: [u8; 8],
    /// The eight 4-bit group scales.
    pub scales: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<BlockIQ2S>() == BLOCK_BYTES);

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

fn check_blocks(weight_len: usize, out_len: usize) -> Result<usize> {
    if !weight_len.is_multiple_of(BLOCK_BYTES) {
        bail!("IQ2_S: weight length {weight_len} is not a multiple of {BLOCK_BYTES}");
    }
    let nb = weight_len / BLOCK_BYTES;
    if out_len != nb * QK_K {
        bail!(
            "IQ2_S: output length {out_len} does not match {nb} super-blocks ({} elements)",
            nb * QK_K
        );
    }
    Ok(nb)
}

/// Decode `len(src) / 82` packed super-blocks into `dst`, bit-identical to
/// `dequantize_row_iq2_s`.
pub fn dequantize(src: &[u8], dst: &mut [f32]) -> Result<()> {
    let nb = check_blocks(src.len(), dst.len())?;
    for b in 0..nb {
        let blk = &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let out = &mut dst[b * QK_K..(b + 1) * QK_K];
        let d = f16_to_f32(rd_u16(blk, 0));
        let grid_idx = &blk[2..34];
        let signs = &blk[34..66];
        let qh = &blk[66..74];
        let sc = &blk[74..82];
        for ib32 in 0..QK_K / 32 {
            let db0 = d * (0.5 + (sc[ib32] & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (sc[ib32] >> 4) as f32) * 0.25;
            for l in 0..4 {
                let dl = if l < 2 { db0 } else { db1 };
                let gi = grid_idx[ib32 * 4 + l] as usize
                    | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                let g = IQ2S_GRID[gi].to_le_bytes();
                let s = signs[ib32 * 4 + l];
                let base = ib32 * 32 + l * 8;
                for j in 0..8 {
                    out[base + j] =
                        dl * g[j] as f32 * if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
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
            "IQ2_S: activation length {} is not a positive multiple of {QK_K}",
            x.len()
        );
    }
    let nb = x.len() / QK_K;
    if dst.len() != nb * Q8K_BYTES {
        bail!(
            "IQ2_S: Q8_K output length {} does not match {nb} blocks ({})",
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

/// Dot product of packed IQ2_S weights against packed Q8_K activations,
/// mirroring `ggml_vec_dot_iq2_s_q8_K_generic`.
pub fn vec_dot_q8_k(w: &[u8], q8: &[u8]) -> Result<f32> {
    if !w.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "IQ2_S: weight length {} is not a multiple of {BLOCK_BYTES}",
            w.len()
        );
    }
    let nb = w.len() / BLOCK_BYTES;
    if q8.len() != nb * Q8K_BYTES {
        bail!(
            "IQ2_S: Q8_K length {} does not match {nb} super-blocks ({})",
            q8.len(),
            nb * Q8K_BYTES
        );
    }
    let mut sumf = 0f32;
    for b in 0..nb {
        let wb = &w[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let qb = &q8[b * Q8K_BYTES..(b + 1) * Q8K_BYTES];
        let d = f16_to_f32(rd_u16(wb, 0)) * rd_f32(qb, 0);
        let grid_idx = &wb[2..34];
        let signs = &wb[34..66];
        let qh = &wb[66..74];
        let sc = &wb[74..82];
        let q8s = &qb[4..260];
        let mut bsum = 0i32;
        for ib32 in 0..QK_K / 32 {
            let ls1 = 1 + 2 * (sc[ib32] & 0xf) as i32;
            let ls2 = 1 + 2 * (sc[ib32] >> 4) as i32;
            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;
            for l in 0..4 {
                let gi = grid_idx[ib32 * 4 + l] as usize
                    | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                let g = IQ2S_GRID[gi].to_le_bytes();
                let s = signs[ib32 * 4 + l];
                let base = ib32 * 32 + l * 8;
                let mut sumi = 0i32;
                for j in 0..8 {
                    let sign = if s & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                    sumi += q8s[base + j] as i8 as i32 * g[j] as i32 * sign;
                }
                if l < 2 {
                    sumi1 += sumi;
                } else {
                    sumi2 += sumi;
                }
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
        }
        sumf += d * bsum as f32;
    }
    Ok(0.125 * sumf)
}

/// Quantize `x` into packed IQ2_S blocks.
///
/// **Not ggml's quantizer** (see the module docs): a plain search over
/// `IQ2S_GRID` with a per-half group-scale search. Valid blocks, not
/// byte-identical.
pub fn quantize_iq2_s(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK_K) {
        bail!(
            "IQ2_S: input length {} is not a positive multiple of {QK_K}",
            x.len()
        );
    }
    let nb = x.len() / QK_K;
    if dst.len() != nb * BLOCK_BYTES {
        bail!(
            "IQ2_S: output length {} does not match {nb} super-blocks ({})",
            dst.len(),
            nb * BLOCK_BYTES
        );
    }
    for b in 0..nb {
        let blk = &mut dst[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        // Largest decoded magnitude is `d * 31/8 * 43`.
        let d_model = if amax > 0.0 {
            amax * 8.0 / (31.0 * 43.0)
        } else {
            0.0
        };
        let d_f16 = half::f16::from_f32(d_model);
        let d_used = d_f16.to_f32();
        let (d_part, rest) = blk.split_at_mut(2);
        d_part.copy_from_slice(&d_f16.to_bits().to_le_bytes());
        let (qs_part, rest) = rest.split_at_mut(64);
        let (grid_idx, signs) = qs_part.split_at_mut(32);
        let (qh, sc) = rest.split_at_mut(8);

        for ib32 in 0..QK_K / 32 {
            let group = &xb[ib32 * 32..ib32 * 32 + 32];
            let mut s4 = [0u8; 2];
            let mut l_grid = [0usize; 4];
            let mut l_sign = [0u8; 4];
            for (half, s4_half) in s4.iter_mut().enumerate() {
                let mut best_err = f32::INFINITY;
                let mut best = (0u8, [0usize; 2], [0u8; 2]);
                for cand in 0..16u32 {
                    let db = d_used * (1 + 2 * cand) as f32 / 8.0;
                    if db == 0.0 {
                        continue;
                    }
                    let mut err = 0f32;
                    let mut gg = [0usize; 2];
                    let mut ss = [0u8; 2];
                    for (li, l) in (half * 2..half * 2 + 2).enumerate() {
                        let sub = &group[l * 8..l * 8 + 8];
                        let mut mask = 0u8;
                        for (j, &v) in sub.iter().enumerate() {
                            if v < 0.0 {
                                mask |= 1 << j;
                            }
                        }
                        let mut target = [0f32; 8];
                        for j in 0..8 {
                            let sign = if mask & KMASK_IQ2XS[j] != 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            target[j] = sign * sub[j] / db;
                        }
                        let (g, e) = best_grid(&target);
                        err += e;
                        gg[li] = g;
                        ss[li] = mask;
                    }
                    if err < best_err {
                        best_err = err;
                        best = (cand as u8, gg, ss);
                    }
                }
                *s4_half = best.0;
                for li in 0..2 {
                    let l = half * 2 + li;
                    l_grid[l] = best.1[li];
                    l_sign[l] = best.2[li];
                }
            }
            sc[ib32] = s4[0] | (s4[1] << 4);
            qh[ib32] = 0;
            for l in 0..4 {
                grid_idx[ib32 * 4 + l] = (l_grid[l] & 0xff) as u8;
                qh[ib32] |= (((l_grid[l] >> 8) & 0x3) as u8) << (2 * l);
                signs[ib32 * 4 + l] = l_sign[l];
            }
        }
    }
    Ok(())
}

/// Nearest `IQ2S_GRID` entry to eight target values (signed; a grid value is in
/// `8..=43`, so a negative target is best served by the smallest entry).
fn best_grid(target: &[f32; 8]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for (g, &entry) in IQ2S_GRID.iter().enumerate() {
        let bytes = entry.to_le_bytes();
        let mut err = 0f32;
        for j in 0..8 {
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
/// `T` must be `#[repr(C)]` with no padding. Only [`BlockIQ2S`] and `BlockQ8K`
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

impl GgmlType for BlockIQ2S {
    const DTYPE: GgmlDType = GgmlDType::IQ2_S;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        // SAFETY: `BlockIQ2S` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes(xs) };
        dequantize(bytes, ys).expect("IQ2_S to_float: block counts match");
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // SAFETY: `BlockIQ2S` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes_mut(ys) };
        quantize_iq2_s(xs, bytes).expect("IQ2_S from_float: block counts match");
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK_K);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_k(w, q).expect("IQ2_S vec_dot: block counts match")
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK_K);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_k(w, q).expect("IQ2_S vec_dot: block counts match")
    }
}

/// Copy packed IQ2_S bytes into `BlockIQ2S` values.
fn blocks_from_bytes(data: &[u8]) -> Result<Vec<BlockIQ2S>> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "IQ2_S: {} bytes is not a multiple of the {BLOCK_BYTES}-byte super-block",
            data.len()
        );
    }
    let mut blocks = vec![
        BlockIQ2S {
            d: 0,
            qs: [0; 64],
            qh: [0; 8],
            scales: [0; 8],
        };
        data.len() / BLOCK_BYTES
    ];
    // SAFETY: `BlockIQ2S` is repr(C) and unpadded (see the size assert).
    unsafe { as_bytes_mut(&mut blocks) }.copy_from_slice(data);
    Ok(blocks)
}

/// Build a CPU `QStorage` holding `data` as IQ2_S super-blocks.
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
        Device::Cuda(_) => bail!("IQ2_S on CUDA needs the `cuda` feature"),
        _ => bail!("IQ2_S is only supported on the CPU and CUDA"),
    }
}

/// `y = W x` for `W` shaped `[rows, cols]` in packed IQ2_S and a single `x` row,
/// the activation quantized to Q8_K once.
#[allow(dead_code)]
pub fn gemv(weights: &[u8], x: &[f32], out: &mut [f32]) -> Result<()> {
    let cols = x.len();
    if cols == 0 || !cols.is_multiple_of(QK_K) {
        bail!("IQ2_S gemv: input length {cols} is not a positive multiple of {QK_K}");
    }
    let row_bytes = (cols / QK_K) * BLOCK_BYTES;
    if !weights.len().is_multiple_of(row_bytes) {
        bail!(
            "IQ2_S gemv: weight length {} is not a multiple of the {row_bytes}-byte row",
            weights.len()
        );
    }
    let rows = weights.len() / row_bytes;
    if out.len() != rows {
        bail!(
            "IQ2_S gemv: output length {} does not match {rows} rows",
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
