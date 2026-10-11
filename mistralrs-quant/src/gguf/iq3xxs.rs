//! CPU kernels for the GGML **IQ3_XXS** weight format (GGML type 18), x86-64.
//!
//! A 256-element super-block: one f16 scale, 64 bytes of grid indices, and 32
//! bytes packing each 32-element group's 4-bit scale and four 7-bit sign
//! indices. The magnitudes come from `IQ3XXS_GRID` (4 values per grid entry);
//! the 16-entry sign masks are `KSIGNS_IQ2XS`. The paired activation is Q8_K.
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

use tables::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

/// Elements per IQ3_XXS super-block.
pub const QK_K: usize = 256;
/// Bytes per packed IQ3_XXS super-block (`d` + 96).
pub const BLOCK_BYTES: usize = 98;
/// Bytes per packed Q8_K activation block.
pub const Q8K_BYTES: usize = 292;

/// One packed IQ3_XXS super-block, byte-compatible with ggml's `block_iq3_xxs`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ3XXS {
    /// Super-block scale, as f16 bits.
    pub d: u16,
    /// 64 grid-index bytes then 32 scale/sign bytes. `[u8; 96]` for two-byte
    /// alignment; the field is exactly `3*QK_K/8`.
    pub qs: [u8; 96],
}

const _: () = assert!(std::mem::size_of::<BlockIQ3XXS>() == BLOCK_BYTES);

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
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
        bail!("IQ3_XXS: weight length {weight_len} is not a multiple of {BLOCK_BYTES}");
    }
    let nb = weight_len / BLOCK_BYTES;
    if out_len != nb * QK_K {
        bail!(
            "IQ3_XXS: output length {out_len} does not match {nb} super-blocks ({} elements)",
            nb * QK_K
        );
    }
    Ok(nb)
}

/// Decode `len(src) / 98` packed super-blocks into `dst`, bit-identical to
/// `dequantize_row_iq3_xxs`.
pub fn dequantize(src: &[u8], dst: &mut [f32]) -> Result<()> {
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
    let qs = &blk[2..BLOCK_BYTES]; // 96 bytes
    let gas = &qs[QK_K / 4..]; // 32 scale/sign bytes
    for ib32 in 0..QK_K / 32 {
        let aux32 = rd_u32(gas, 4 * ib32);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l] as usize].to_le_bytes();
            let g2 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l + 1] as usize].to_le_bytes();
            let base = ib32 * 32 + l * 8;
            for j in 0..4 {
                out[base + j] = db
                    * g1[j] as f32
                    * if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                out[base + 4 + j] = db
                    * g2[j] as f32
                    * if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
            }
        }
    }
}

/// Quantize `x` into packed Q8_K blocks, bit-identical to
/// `quantize_row_q8_K_ref` (bsums are zeroed for an all-zero block).
#[allow(dead_code)]
pub fn quantize_row_q8_k(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK_K) {
        bail!(
            "IQ3_XXS: activation length {} is not a positive multiple of {QK_K}",
            x.len()
        );
    }
    let nb = x.len() / QK_K;
    if dst.len() != nb * Q8K_BYTES {
        bail!(
            "IQ3_XXS: Q8_K output length {} does not match {nb} blocks ({})",
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

/// Dot product of packed IQ3_XXS weights against packed Q8_K activations,
/// mirroring `ggml_vec_dot_iq3_xxs_q8_K_generic`.
pub fn vec_dot_q8_k(w: &[u8], q8: &[u8]) -> Result<f32> {
    if !w.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "IQ3_XXS: weight length {} is not a multiple of {BLOCK_BYTES}",
            w.len()
        );
    }
    let nb = w.len() / BLOCK_BYTES;
    if q8.len() != nb * Q8K_BYTES {
        bail!(
            "IQ3_XXS: Q8_K length {} does not match {nb} super-blocks ({})",
            q8.len(),
            nb * Q8K_BYTES
        );
    }
    let mut sumf = 0f32;
    for b in 0..nb {
        let wb = &w[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let qb = &q8[b * Q8K_BYTES..(b + 1) * Q8K_BYTES];
        let d = f16_to_f32(rd_u16(wb, 0)) * rd_f32(qb, 0);
        let qs = &wb[2..BLOCK_BYTES];
        let gas = &qs[QK_K / 4..];
        let q8s = &qb[4..260];
        let mut bsum = 0i32;
        for ib32 in 0..QK_K / 32 {
            let aux32 = rd_u32(gas, 4 * ib32);
            let ls = 2 * (aux32 >> 28) + 1;
            let mut sumi = 0i32;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l + 1] as usize].to_le_bytes();
                for j in 0..4 {
                    let s0 = if signs & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                    let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1
                    } else {
                        1
                    };
                    sumi += g1[j] as i32 * q8s[ib32 * 32 + l * 8 + j] as i8 as i32 * s0;
                    sumi += g2[j] as i32 * q8s[ib32 * 32 + l * 8 + 4 + j] as i8 as i32 * s1;
                }
            }
            bsum += sumi * ls as i32;
        }
        sumf += d * bsum as f32;
    }
    Ok(0.25 * sumf)
}

/// Quantize `x` into packed IQ3_XXS blocks.
///
/// **Not ggml's quantizer.** ggml searches a precomputed 12-bit grid map and its
/// neighbours; this does a direct search over `IQ3XXS_GRID` and picks the group
/// scale that minimises squared error. The blocks are valid IQ3_XXS, and the
/// result round-trips with a small error; they are not byte-identical to
/// `quantize_row_iq3_xxs_ref`. Only the load path matters here, and it never
/// quantizes.
pub fn quantize_iq3_xxs(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK_K) {
        bail!(
            "IQ3_XXS: input length {} is not a positive multiple of {QK_K}",
            x.len()
        );
    }
    let nb = x.len() / QK_K;
    if dst.len() != nb * BLOCK_BYTES {
        bail!(
            "IQ3_XXS: output length {} does not match {nb} super-blocks ({})",
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

/// Nearest grid index to four target values, which may be negative (a decoded
/// sign the raw value disagrees with is best served by a zero grid value).
fn best_grid(target: &[f32; 4]) -> u8 {
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for (g, &entry) in IQ3XXS_GRID.iter().enumerate() {
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
    best as u8
}

fn quantize_block(xb: &[f32], out: &mut [u8]) {
    let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
    // Peak decodes as `d * ls/4 * 3`; put that at `amax` with the largest `ls`.
    let d_model = if amax > 0.0 {
        amax * 4.0 / (31.0 * 62.0)
    } else {
        0.0
    };
    let d_f16 = half::f16::from_f32(d_model);
    let d_used = d_f16.to_f32();
    out[..2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
    let qs = &mut out[2..BLOCK_BYTES];
    let (grid_part, gas) = qs.split_at_mut(QK_K / 4);

    for ib32 in 0..QK_K / 32 {
        let group = &xb[ib32 * 32..ib32 * 32 + 32];
        // Pick the 4-bit group scale that minimises error.
        let mut best_s4 = 0u32;
        let mut best_signs = [0u8; 4];
        let mut best_grids = [(0u8, 0u8); 4];
        let mut best_err = f32::INFINITY;
        for s4 in 0..16u32 {
            let db = d_used * (2 * s4 + 1) as f32 / 4.0;
            if db == 0.0 {
                continue;
            }
            let mut err = 0f32;
            let mut signs = [0u8; 4];
            let mut grids = [(0u8, 0u8); 4];
            for l in 0..4 {
                let sub = &group[l * 8..l * 8 + 8];
                let mut mask = 0u8;
                for (j, &v) in sub.iter().enumerate() {
                    if v < 0.0 {
                        mask |= 1 << j;
                    }
                }
                let s = sign_index(mask);
                let decoded = KSIGNS_IQ2XS[s as usize];
                signs[l] = s;
                // Target the signed value the decoder will produce, so an
                // element whose decoded sign disagrees is best served by a 0.
                let mut t1 = [0f32; 4];
                let mut t2 = [0f32; 4];
                for j in 0..4 {
                    let s0 = if decoded & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s1 = if decoded & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    t1[j] = s0 * sub[j] / db;
                    t2[j] = s1 * sub[j + 4] / db;
                }
                let g1 = best_grid(&t1);
                let g2 = best_grid(&t2);
                grids[l] = (g1, g2);
                let b1 = IQ3XXS_GRID[g1 as usize].to_le_bytes();
                let b2 = IQ3XXS_GRID[g2 as usize].to_le_bytes();
                for j in 0..4 {
                    let s0 = if decoded & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s1 = if decoded & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let e0 = db * b1[j] as f32 * s0 - sub[j];
                    let e1 = db * b2[j] as f32 * s1 - sub[j + 4];
                    err += e0 * e0 + e1 * e1;
                }
            }
            if err < best_err {
                best_err = err;
                best_s4 = s4;
                best_signs = signs;
                best_grids = grids;
            }
        }
        let mut aux32 = best_s4 << 28;
        for l in 0..4 {
            aux32 |= (best_signs[l] as u32) << (7 * l);
            grid_part[ib32 * 8 + 2 * l] = best_grids[l].0;
            grid_part[ib32 * 8 + 2 * l + 1] = best_grids[l].1;
        }
        gas[4 * ib32..4 * ib32 + 4].copy_from_slice(&aux32.to_le_bytes());
    }
}

/// Map an 8-bit sign mask to the nearest representable 7-bit index.
///
/// The representable masks are `KSIGNS_IQ2XS`; the 8th bit is redundant, so a
/// mask that does not match exactly picks the fewest-flips entry.
fn sign_index(mask: u8) -> u8 {
    static REVERSE: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    let table = REVERSE.get_or_init(|| {
        let mut rev = [0u8; 256];
        for (m, slot) in rev.iter_mut().enumerate() {
            let mut best = 0u8;
            let mut best_flips = u32::MAX;
            for (s, &decoded) in KSIGNS_IQ2XS.iter().enumerate() {
                let flips = (m as u8 ^ decoded).count_ones();
                if flips < best_flips {
                    best_flips = flips;
                    best = s as u8;
                }
            }
            *slot = best;
        }
        rev
    });
    table[mask as usize]
}

/// View a `#[repr(C)]`, padding-free block slice as its packed bytes.
///
/// # Safety
///
/// `T` must be `#[repr(C)]` with no padding. Only [`BlockIQ3XXS`] and `BlockQ8K`
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

impl GgmlType for BlockIQ3XXS {
    const DTYPE: GgmlDType = GgmlDType::IQ3_XXS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        // SAFETY: `BlockIQ3XXS` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes(xs) };
        dequantize(bytes, ys).expect("IQ3_XXS to_float: block counts match");
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // SAFETY: `BlockIQ3XXS` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes_mut(ys) };
        quantize_iq3_xxs(xs, bytes).expect("IQ3_XXS from_float: block counts match");
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK_K);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_k(w, q).expect("IQ3_XXS vec_dot: block counts match")
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK_K);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_k(w, q).expect("IQ3_XXS vec_dot: block counts match")
    }
}

/// Copy packed IQ3_XXS bytes into `BlockIQ3XXS` values.
fn blocks_from_bytes(data: &[u8]) -> Result<Vec<BlockIQ3XXS>> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "IQ3_XXS: {} bytes is not a multiple of the {BLOCK_BYTES}-byte super-block",
            data.len()
        );
    }
    let mut blocks = vec![BlockIQ3XXS { d: 0, qs: [0; 96] }; data.len() / BLOCK_BYTES];
    // SAFETY: `BlockIQ3XXS` is repr(C) and unpadded (see the size assert).
    unsafe { as_bytes_mut(&mut blocks) }.copy_from_slice(data);
    Ok(blocks)
}

/// Build a CPU `QStorage` holding `data` as IQ3_XXS super-blocks.
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
        Device::Cuda(_) => bail!("IQ3_XXS on CUDA needs the `cuda` feature"),
        _ => bail!("IQ3_XXS is only supported on the CPU and CUDA"),
    }
}

/// `y = W x` for `W` shaped `[rows, cols]` in packed IQ3_XXS and a single `x`
/// row, the activation quantized to Q8_K once.
#[allow(dead_code)]
pub fn gemv(weights: &[u8], x: &[f32], out: &mut [f32]) -> Result<()> {
    let cols = x.len();
    if cols == 0 || !cols.is_multiple_of(QK_K) {
        bail!("IQ3_XXS gemv: input length {cols} is not a positive multiple of {QK_K}");
    }
    let row_bytes = (cols / QK_K) * BLOCK_BYTES;
    if !weights.len().is_multiple_of(row_bytes) {
        bail!(
            "IQ3_XXS gemv: weight length {} is not a multiple of the {row_bytes}-byte row",
            weights.len()
        );
    }
    let rows = weights.len() / row_bytes;
    if out.len() != rows {
        bail!(
            "IQ3_XXS gemv: output length {} does not match {rows} rows",
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
