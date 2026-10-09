//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use candle_core::{
    quantized::{GgmlType, QMatMul, QTensor},
    Result, Tensor,
};
use candle_nn::Linear;
use std::sync::Arc;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Perform indexed MoE forward pass on a QTensor by dequantizing and using UnquantLinear.
///
/// # Arguments
/// * `qtensor` - The quantized weight tensor [num_experts, n, k]
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn qtensor_indexed_moe_forward(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    // Repacked per-expert gemv path (aarch64), else a routed-expert gemv over the stacked
    // blocks; falls back to dequantize-and-gather only for layouts and types neither serves. Normalize the metal/cpu 4D/5D input
    // shapes to the (tokens, x_t, hidden) form the kernel expects.
    {
        let (x3, ids2, out_shape): (Tensor, Tensor, Option<Vec<usize>>) = match *x.dims() {
            [b, s, xt, h] => {
                let (ib, is, t) = ids.dims3()?;
                if ib == b && is == s {
                    (
                        x.reshape((b * s, xt, h))?,
                        ids.reshape((b * s, t))?,
                        Some(vec![b, s, t, 0]),
                    )
                } else {
                    (x.clone(), ids.clone(), None)
                }
            }
            [b, s, 1, 1, h] => {
                let (ib, is, t) = ids.dims3()?;
                if ib == b && is == s {
                    (
                        x.reshape((b * s, 1, h))?,
                        ids.reshape((b * s, t))?,
                        Some(vec![b, s, t, 0]),
                    )
                } else {
                    (x.clone(), ids.clone(), None)
                }
            }
            [_, _, _] => (x.clone(), ids.clone(), Some(vec![])),
            _ => (x.clone(), ids.clone(), None),
        };
        if let Some(shape) = out_shape.filter(|_| x3.rank() == 3 && ids2.rank() == 2) {
            let out = match qtensor.indexed_gemv(&x3, &ids2)? {
                Some(out) => Some(out),
                None => routed_gemv(qtensor, &x3, &ids2)?,
            };
            if let Some(out) = out {
                return if shape.is_empty() {
                    Ok(out)
                } else {
                    let n_out = out.dim(2)?;
                    out.reshape((shape[0], shape[1], shape[2], n_out))
                };
            }
        }
    }

    let device = x.device();

    // Dequantize all weights to f32
    let weights = qtensor.dequantize(device)?;

    // Create an UnquantLinear and use its gather_forward
    let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;

    unquant.gather_forward(x, ids)
}

/// Quantized gemv over only the routed experts, reading the stacked weights in place.
///
/// `indexed_gemv` has a packed fast path on aarch64 only; elsewhere the fallback above dequantizes
/// the whole `[E, N, K]` stack to f32 on every call (~4x the quantized size, per layer). This
/// quantizes each activation row once to the weight type's vec-dot format and dots it with the
/// rows of the experts it was routed to, as `QMatMul` does for a single expert.
///
/// Returns `None` for shapes and types it does not serve (the caller falls back).
fn routed_gemv(qtensor: &QTensor, x: &Tensor, ids: &Tensor) -> Result<Option<Tensor>> {
    use candle_core::quantized::{k_quants::*, GgmlDType};

    if !x.device().is_cpu() || !qtensor.device().is_cpu() {
        return Ok(None);
    }
    let (Ok((n_experts, n, k)), Ok((tokens, x_t, xk)), Ok((ids_t, topk))) =
        (qtensor.shape().dims3(), x.dims3(), ids.dims2())
    else {
        return Ok(None);
    };
    if xk != k || ids_t != tokens || (x_t != 1 && x_t != topk) {
        return Ok(None);
    }
    let data = qtensor.data()?;
    let xs = x
        .to_dtype(candle_core::DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let ids = ids
        .to_dtype(candle_core::DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    if let Some(&bad) = ids.iter().find(|&&e| e as usize >= n_experts) {
        candle_core::bail!("expert id {bad} out of range for {n_experts} experts");
    }
    if ids.len() != tokens * topk {
        return Ok(None);
    }
    let dst = match qtensor.dtype() {
        GgmlDType::Q4_0 => gemv::<BlockQ4_0>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q4_1 => gemv::<BlockQ4_1>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q5_0 => gemv::<BlockQ5_0>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q5_1 => gemv::<BlockQ5_1>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q8_0 => gemv::<BlockQ8_0>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q2K => gemv::<BlockQ2K>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q3K => gemv::<BlockQ3K>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q4K => gemv::<BlockQ4K>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q5K => gemv::<BlockQ5K>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q6K => gemv::<BlockQ6K>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        _ => return Ok(None),
    };
    let out = Tensor::from_vec(dst, (tokens, topk, n), x.device())?;
    Ok(Some(out.to_dtype(x.dtype())?))
}

/// Output rows per rayon task in `gemv`: enough to amortize a task, few enough that one expert
/// spreads over every thread (decode rows are 640-2560 long here).
const ROWS_PER_TASK: usize = 32;

/// `dst[t, j, :] = W[ids[t, j]] · x[t, j or 0]` with `W` the `[E, n, k]` blocks of type `T`.
fn gemv<T: GgmlType>(
    data: &[u8],
    xs: &[f32],
    ids: &[u32],
    (tokens, x_t, topk, n, k): (usize, usize, usize, usize, usize),
) -> Result<Vec<f32>> {
    use candle_core::quantized::GgmlType as _;
    use rayon::prelude::*;

    if k % T::BLCK_SIZE != 0 || k % T::VecDotType::BLCK_SIZE != 0 {
        candle_core::bail!("row length {k} is not a multiple of the block size");
    }
    let row_blocks = k / T::BLCK_SIZE;
    let block_bytes = std::mem::size_of::<T>();
    let expert_blocks = n * row_blocks;
    let n_experts = data.len() / (expert_blocks * block_bytes);
    if data.len() != n_experts * expert_blocks * block_bytes
        || ids.iter().any(|&e| e as usize >= n_experts)
        || data.as_ptr().align_offset(std::mem::align_of::<T>()) != 0
    {
        candle_core::bail!("expert stack bytes do not match its shape");
    }
    // SAFETY: `T` is a `repr(C)` ggml block; the length and alignment were checked above.
    let blocks: &[T] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<T>(), data.len() / block_bytes) };

    // Each distinct activation row, quantized once for the vec-dot.
    let vd_row = k / T::VecDotType::BLCK_SIZE;
    let mut xq = vec![T::VecDotType::zeros(); tokens * x_t * vd_row];
    xq.par_chunks_mut(vd_row)
        .zip(xs.par_chunks(k))
        .for_each(|(dst, src)| T::VecDotType::from_float(src, dst));

    let mut dst = vec![0f32; tokens * topk * n];
    // Rows are split too: a decode step often routes only a couple of experts here (the rest
    // run elsewhere), which would otherwise leave all but a couple of threads idle.
    dst.par_chunks_mut(n).enumerate().for_each(|(slot, out)| {
        let (t, j) = (slot / topk, slot % topk);
        let xrow = t * x_t + if x_t == 1 { 0 } else { j };
        let x = &xq[xrow * vd_row..(xrow + 1) * vd_row];
        let w = &blocks[ids[slot] as usize * expert_blocks..][..expert_blocks];
        out.par_chunks_mut(ROWS_PER_TASK)
            .zip(w.par_chunks(ROWS_PER_TASK * row_blocks))
            .for_each(|(out, w)| {
                for (o, row) in out.iter_mut().zip(w.chunks_exact(row_blocks)) {
                    *o = T::vec_dot(k, row, x);
                }
            });
    });
    Ok(dst)
}

/// Perform indexed MoE forward pass on a QMatMul.
///
/// This is the main entry point for CPU/Metal GGUF quantized MoE forward.
///
/// # Arguments
/// * `qmatmul` - The quantized weight matrix
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn cpu_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // For non-quantized tensors, use UnquantLinear directly
            let unquant =
                UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(t.clone(), None)))?;
            unquant.gather_forward(x, ids)
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{
        quantized::{GgmlDType, QMatMul, QTensor},
        Device, IndexOp, Module, Tensor,
    };

    use super::routed_gemv;

    /// The public entry point takes the model's 4D (`[b, s, x_t, h]`) activations, in BF16, and
    /// must reach the routed gemv rather than the dequantize fallback.
    #[test]
    fn indexed_moe_forward_serves_4d_bf16_input() {
        let (e, n, k) = (4, 16, 256);
        let w: Vec<f32> = (0..e * n * k)
            .map(|i| ((i as f32) * 0.37).sin() * 0.5)
            .collect();
        let w = Tensor::from_vec(w, (e, n, k), &Device::Cpu).unwrap();
        let q = std::sync::Arc::new(QTensor::quantize(&w, GgmlDType::Q4K).unwrap());
        let x: Vec<f32> = (0..2 * k).map(|i| ((i as f32) * 0.11).cos()).collect();
        let x = Tensor::from_vec(x, (1, 2, 1, k), &Device::Cpu)
            .unwrap()
            .to_dtype(candle_core::DType::BF16)
            .unwrap();
        let ids = Tensor::new(&[[[1u32, 3], [0, 2]]], &Device::Cpu).unwrap();
        let out = super::qtensor_indexed_moe_forward(&q, &x, &ids).unwrap();
        assert_eq!(out.dims(), &[1, 2, 2, n]);
        assert_eq!(out.dtype(), candle_core::DType::BF16);
    }

    /// Each routed slot must equal running its expert's own `QMatMul` on the same row.
    #[test]
    fn routed_gemv_matches_per_expert_qmatmul() {
        let (e, n, k, tokens, topk) = (4, 24, 256, 3, 2);
        let w: Vec<f32> = (0..e * n * k)
            .map(|i| ((i as f32) * 0.37).sin() * 0.5)
            .collect();
        let w = Tensor::from_vec(w, (e, n, k), &Device::Cpu).unwrap();
        let ids = Tensor::new(&[[0u32, 2], [3, 1], [2, 2]], &Device::Cpu).unwrap();
        for dtype in [
            GgmlDType::Q4_0,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q6K,
        ] {
            let q = QTensor::quantize(&w, dtype).unwrap();
            for x_t in [1, topk] {
                let x: Vec<f32> = (0..tokens * x_t * k)
                    .map(|i| ((i as f32) * 0.11).cos())
                    .collect();
                let x = Tensor::from_vec(x, (tokens, x_t, k), &Device::Cpu).unwrap();
                let got = routed_gemv(&q, &x, &ids).unwrap().unwrap();
                assert_eq!(got.dims(), &[tokens, topk, n]);
                let ids_v = ids.to_vec2::<u32>().unwrap();
                for (t, row_ids) in ids_v.iter().enumerate() {
                    for (j, &ex) in row_ids.iter().enumerate() {
                        let ex = ex as usize;
                        // fresh buffer: `quantize` reads the whole backing storage, ignoring a view's offset
                        let wex = Tensor::from_vec(
                            w.i(ex)
                                .unwrap()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap(),
                            (n, k),
                            &Device::Cpu,
                        )
                        .unwrap();
                        let one = QTensor::quantize(&wex, dtype).unwrap();
                        let row = x
                            .i((t, if x_t == 1 { 0 } else { j }))
                            .unwrap()
                            .unsqueeze(0)
                            .unwrap();
                        let want = QMatMul::from_qtensor(one).unwrap().forward(&row).unwrap();
                        let diff = (got.i((t, j)).unwrap().unsqueeze(0).unwrap() - want)
                            .unwrap()
                            .abs()
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .max(0)
                            .unwrap()
                            .to_scalar::<f32>()
                            .unwrap();
                        assert!(
                            diff < 1e-4,
                            "{dtype:?} x_t={x_t} slot ({t},{j}) diverges by {diff}"
                        );
                    }
                }
            }
        }
    }
}
