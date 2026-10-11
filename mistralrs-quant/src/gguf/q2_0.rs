//! CPU kernels for the GGML **Q2_0** weight format (GGML type 42), x86-64.
//!
//! Q2_0 is a plain 2-bit format: 64 elements per block with one f16 scale, no
//! codebook, and the four 2-bit codes decode to `-1, 0, +1, +2` times the scale.
//! The paired activation is Q8_0 (two 32-element blocks per weight block). It is
//! the simplest of the i-quants the published Qwen3.8-Flash-Next GGUFs use.
//!
//! # Layout
//!
//! ```text
//! offset  0: d    u16 (f16 bits)  block scale
//! offset  2: qs   [u8; 16]        4 two-bit codes per byte, low bits first
//! ```
//!
//! Element `j` is code `(qs[j/4] >> (2 * (j % 4))) & 3`, decoded as `code - 1`.

use candle_core::{
    bail,
    quantized::{
        k_quants::{BlockQ8_0, GgmlType},
        GgmlDType, QStorage,
    },
    Device, Result,
};

/// Elements per Q2_0 block.
pub const QK2_0: usize = 64;
/// Bytes per packed Q2_0 block.
pub const BLOCK_BYTES: usize = 18;
/// Bytes per packed Q8_0 activation block.
pub const Q8_0_BYTES: usize = 34;

/// One packed Q2_0 block, byte-compatible with ggml's `block_q2_0`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ2_0 {
    /// Block scale, as f16 bits. Note: ggml stores the group's max absolute
    /// value, not a divided scale.
    pub d: u16,
    /// 16 bytes of two-bit codes.
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ2_0>() == BLOCK_BYTES);

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

/// Validate a `(weights, output)` pair and return the block count.
fn check_blocks(weight_len: usize, out_len: usize) -> Result<usize> {
    if !weight_len.is_multiple_of(BLOCK_BYTES) {
        bail!("Q2_0: weight length {weight_len} is not a multiple of {BLOCK_BYTES}");
    }
    let nb = weight_len / BLOCK_BYTES;
    if out_len != nb * QK2_0 {
        bail!(
            "Q2_0: output length {out_len} does not match {nb} blocks ({} elements)",
            nb * QK2_0
        );
    }
    Ok(nb)
}

/// Decode `len(src) / 18` packed blocks into `dst`, bit-identical to
/// `dequantize_row_q2_0`.
///
/// Scalar only: ggml has no SIMD kernel for Q2_0 either (its CPU dot is the
/// generic one).
pub fn dequantize(src: &[u8], dst: &mut [f32]) -> Result<()> {
    let nb = check_blocks(src.len(), dst.len())?;
    for b in 0..nb {
        let blk = &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let out = &mut dst[b * QK2_0..(b + 1) * QK2_0];
        let d = f16_to_f32(rd_u16(blk, 0));
        let qs = &blk[2..BLOCK_BYTES];
        for j in 0..QK2_0 {
            let code = (qs[j / 4] >> (2 * (j % 4))) & 3;
            out[j] = (code as i32 - 1) as f32 * d;
        }
    }
    Ok(())
}

/// Quantize `x` into packed Q8_0 blocks, bit-identical to
/// `quantize_row_q8_0_ref`.
///
/// `candle-core`'s `BlockQ8_0::from_float` does the same for the wired-up path;
/// this exists so the kernel and its tests do not depend on private candle
/// fields.
#[allow(dead_code)]
pub fn quantize_row_q8_0(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(32) {
        bail!(
            "Q2_0: activation length {} is not a positive multiple of 32",
            x.len()
        );
    }
    let nb = x.len() / 32;
    if dst.len() != nb * Q8_0_BYTES {
        bail!(
            "Q2_0: Q8_0 output length {} does not match {nb} blocks ({})",
            dst.len(),
            nb * Q8_0_BYTES
        );
    }
    for b in 0..nb {
        let xb = &x[b * 32..(b + 1) * 32];
        let out = &mut dst[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES];
        let mut amax = 0f32;
        for &v in xb {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for j in 0..32 {
            out[2 + j] = (xb[j] * id).round() as i8 as u8;
        }
    }
    Ok(())
}

/// Dot product of packed Q2_0 weights against packed Q8_0 activations.
///
/// Scalar only (see [`dequantize`]).
pub fn vec_dot_q8_0(w: &[u8], q8: &[u8]) -> Result<f32> {
    let nb = check_dot(w, q8)?;
    let mut sumf = 0f32;
    for b in 0..nb {
        sumf += sum_block(
            &w[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
            &q8[b * 2 * Q8_0_BYTES..(b + 1) * 2 * Q8_0_BYTES],
        );
    }
    Ok(sumf)
}

fn check_dot(w: &[u8], q8: &[u8]) -> Result<usize> {
    if !w.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "Q2_0: weight length {} is not a multiple of {BLOCK_BYTES}",
            w.len()
        );
    }
    let nb = w.len() / BLOCK_BYTES;
    if q8.len() != nb * 2 * Q8_0_BYTES {
        bail!(
            "Q2_0: Q8_0 length {} does not match {nb} blocks ({})",
            q8.len(),
            nb * 2 * Q8_0_BYTES
        );
    }
    Ok(nb)
}

/// One Q2_0 block (64 weights) against two Q8_0 blocks (2 * 32 activations).
#[inline]
fn sum_block(wb: &[u8], q8: &[u8]) -> f32 {
    let d0 = f16_to_f32(rd_u16(wb, 0));
    let qs = &wb[2..BLOCK_BYTES];
    let mut sumf = 0f32;
    for k in 0..2 {
        let qb = &q8[k * Q8_0_BYTES..(k + 1) * Q8_0_BYTES];
        let d1 = f16_to_f32(rd_u16(qb, 0));
        let qy = &qb[2..Q8_0_BYTES];
        let mut sumi = 0i32;
        for b in 0..8 {
            let byte = qs[k * 8 + b];
            sumi += ((byte & 3) as i32 - 1) * qy[b * 4] as i8 as i32;
            sumi += (((byte >> 2) & 3) as i32 - 1) * qy[b * 4 + 1] as i8 as i32;
            sumi += (((byte >> 4) & 3) as i32 - 1) * qy[b * 4 + 2] as i8 as i32;
            sumi += (((byte >> 6) & 3) as i32 - 1) * qy[b * 4 + 3] as i8 as i32;
        }
        sumf += d0 * d1 * sumi as f32;
    }
    sumf
}

/// `y = W x` for `W` shaped `[rows, cols]` in packed Q2_0 and a single `x` row.
/// The activation row is quantized to Q8_0 once and reused for every row.
///
/// Standalone entry point for tests and benchmarks; the engine reaches the
/// kernels through `crate::gguf::cpu::routed_gemv` and `GgmlType` instead.
#[allow(dead_code)]
pub fn gemv(weights: &[u8], x: &[f32], out: &mut [f32]) -> Result<()> {
    let cols = x.len();
    if cols == 0 || !cols.is_multiple_of(QK2_0) {
        bail!("Q2_0 gemv: input length {cols} is not a positive multiple of {QK2_0}");
    }
    let row_bytes = (cols / QK2_0) * BLOCK_BYTES;
    if !weights.len().is_multiple_of(row_bytes) {
        bail!(
            "Q2_0 gemv: weight length {} is not a multiple of the {row_bytes}-byte row",
            weights.len()
        );
    }
    let rows = weights.len() / row_bytes;
    if out.len() != rows {
        bail!(
            "Q2_0 gemv: output length {} does not match {rows} rows",
            out.len()
        );
    }
    let mut q8 = vec![0u8; (cols / QK2_0) * 2 * Q8_0_BYTES];
    quantize_row_q8_0(x, &mut q8)?;
    for (row, o) in out.iter_mut().enumerate() {
        *o = vec_dot_q8_0(&weights[row * row_bytes..(row + 1) * row_bytes], &q8)?;
    }
    Ok(())
}

/// Quantize `x` into packed Q2_0 blocks, bit-identical to
/// `quantize_row_q2_0_ref`. Also the convert-to-another-variant fallback.
pub fn quantize_q2_0(x: &[f32], dst: &mut [u8]) -> Result<()> {
    if x.is_empty() || !x.len().is_multiple_of(QK2_0) {
        bail!(
            "Q2_0: input length {} is not a positive multiple of {QK2_0}",
            x.len()
        );
    }
    let nb = x.len() / QK2_0;
    if dst.len() != nb * BLOCK_BYTES {
        bail!(
            "Q2_0: output length {} does not match {nb} blocks ({})",
            dst.len(),
            nb * BLOCK_BYTES
        );
    }
    for b in 0..nb {
        let xb = &x[b * QK2_0..(b + 1) * QK2_0];
        let out = &mut dst[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let mut amax = 0f32;
        for &v in xb {
            amax = amax.max(v.abs());
        }
        let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };
        out[..2].copy_from_slice(&half::f16::from_f32(amax).to_bits().to_le_bytes());
        let mut qs = [0u8; 16];
        for (j, &v) in xb.iter().enumerate() {
            let q = ((v * id).round() as i32 + 1).clamp(0, 3) as u8;
            qs[j / 4] |= q << (2 * (j % 4));
        }
        out[2..BLOCK_BYTES].copy_from_slice(&qs);
    }
    Ok(())
}

/// View a `#[repr(C)]`, padding-free block slice as its packed bytes.
///
/// # Safety
///
/// `T` must be `#[repr(C)]` with no padding. Only [`BlockQ2_0`] and `BlockQ8_0`
/// (both asserted padding-free by candle) are ever passed.
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

impl GgmlType for BlockQ2_0 {
    const DTYPE: GgmlDType = GgmlDType::Q2_0;
    const BLCK_SIZE: usize = QK2_0;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        // SAFETY: `BlockQ2_0` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes(xs) };
        dequantize(bytes, ys).expect("Q2_0 to_float: block counts match");
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // SAFETY: `BlockQ2_0` is repr(C) and unpadded (see the size assert).
        let bytes = unsafe { as_bytes_mut(ys) };
        quantize_q2_0(xs, bytes).expect("Q2_0 from_float: block counts match");
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK2_0);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_0(w, q).expect("Q2_0 vec_dot: block counts match")
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert_eq!(n, xs.len() * QK2_0);
        // SAFETY: both block types are repr(C) and unpadded.
        let (w, q) = unsafe { (as_bytes(xs), as_bytes(ys)) };
        vec_dot_q8_0(w, q).expect("Q2_0 vec_dot: block counts match")
    }
}

/// Copy packed Q2_0 bytes into `BlockQ2_0` values.
fn blocks_from_bytes(data: &[u8]) -> Result<Vec<BlockQ2_0>> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        bail!(
            "Q2_0: {} bytes is not a multiple of the {BLOCK_BYTES}-byte block",
            data.len()
        );
    }
    let mut blocks = vec![BlockQ2_0 { d: 0, qs: [0; 16] }; data.len() / BLOCK_BYTES];
    // SAFETY: `BlockQ2_0` is repr(C) and unpadded (see the size assert).
    unsafe { as_bytes_mut(&mut blocks) }.copy_from_slice(data);
    Ok(blocks)
}

/// Build a CPU `QStorage` holding `data` as Q2_0 blocks.
///
/// `candle-core`'s `GgmlDType::Q2_0` has no block type of its own (its
/// `from_data` is a stub), so the GGUF loader calls this instead of
/// `QStorage::from_data`.
pub fn cpu_storage(data: &[u8]) -> Result<QStorage> {
    Ok(QStorage::Cpu(Box::new(blocks_from_bytes(data)?)))
}

/// Build the storage for `data` on `device`.
///
/// CPU keeps the packed blocks as-is ([`cpu_storage`]). CUDA uploads the same
/// bytes through candle's `cuda::load_quantized`, which only needs the block's
/// `DTYPE` and size; the kernels are the fork's.
pub fn storage_for_device(data: &[u8], device: &Device) -> Result<QStorage> {
    match device {
        Device::Cpu => cpu_storage(data),
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            let blocks = blocks_from_bytes(data)?;
            candle_core::quantized::cuda::load_quantized(dev, &blocks)
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => bail!("Q2_0 on CUDA needs the `cuda` feature"),
        _ => bail!("Q2_0 is only supported on the CPU and CUDA"),
    }
}

#[cfg(test)]
mod tests;
