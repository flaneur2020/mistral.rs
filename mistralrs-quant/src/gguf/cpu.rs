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

    use super::iq2s::BlockIQ2S;
    use super::iq3s::BlockIQ3S;
    use super::iq3xxs::BlockIQ3XXS;
    use super::iq4nl::BlockIQ4NL;
    use super::iq4xs::BlockIQ4XS;
    use super::q2_0::BlockQ2_0;

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
        GgmlDType::IQ4XS => gemv::<BlockIQ4XS>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::IQ4NL => gemv::<BlockIQ4NL>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::Q2_0 => gemv::<BlockQ2_0>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::IQ3_XXS => gemv::<BlockIQ3XXS>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::IQ2_S => gemv::<BlockIQ2S>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
        GgmlDType::IQ3_S => gemv::<BlockIQ3S>(&data, &xs, &ids, (tokens, x_t, topk, n, k))?,
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

    // Measurement hook (IQ_DBG_UNION=1): per call, how many expert slots ran here
    // versus how many distinct experts they touch. A verify step that is not
    // grouped by expert reads one set per (token, topk) slot; grouping by expert
    // would read only the distinct ones.
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var_os("IQ_DBG_UNION").is_some()) {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static N: AtomicUsize = AtomicUsize::new(0);
            let i = N.fetch_add(1, Ordering::Relaxed);
            if i < 2000 {
                let distinct = ids
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                println!(
                    "IQ_DBG_UNION #{i} slots={} distinct={} tokens={tokens} x_t={x_t} topk={topk} n={n} k={k}",
                    ids.len(),
                    distinct
                );
            }
        }
    }

    // Each distinct activation row, quantized once for the vec-dot.
    let vd_row = k / T::VecDotType::BLCK_SIZE;
    let mut xq = vec![T::VecDotType::zeros(); tokens * x_t * vd_row];
    xq.par_chunks_mut(vd_row)
        .zip(xs.par_chunks(k))
        .for_each(|(dst, src)| T::VecDotType::from_float(src, dst));

    let mut dst = vec![0f32; tokens * topk * n];

    // Slots that share both an expert and an activation row produce bit-identical outputs, so
    // they only need one dot product between them. That happens for 21% of an MTP verify
    // batch's slots (its three tokens often share one activation row *and* route to the same
    // experts), where the old per-slot loop recomputed the same dot. Grouping by the pair
    // rather than by the expert alone is what makes the saving real: deduping the weight
    // *reads* only, as an earlier attempt did, still executed every slot.
    use std::collections::HashMap;
    let xrow_of = |slot: usize| -> usize {
        let (t, j) = (slot / topk, slot % topk);
        t * x_t + if x_t == 1 { 0 } else { j }
    };
    let mut by_pair: HashMap<(u32, usize), Vec<u32>> = HashMap::with_capacity(ids.len());
    for (slot, &expert) in ids.iter().enumerate() {
        by_pair
            .entry((expert, xrow_of(slot)))
            .or_default()
            .push(slot as u32);
    }
    let mut groups: Vec<((u32, usize), Vec<u32>)> = by_pair.into_iter().collect();
    groups.sort_unstable_by_key(|(key, _)| *key); // walk the expert stack in order

    // One task per (group, row block): enough tasks to fill the threads even when a decode
    // step routes only a couple of experts here.
    let tasks: Vec<((u32, usize), usize, &[u32])> = groups
        .iter()
        .flat_map(|(key, slots)| {
            (0..n.div_ceil(ROWS_PER_TASK)).map(move |c| (*key, c, slots.as_slice()))
        })
        .collect();

    // SAFETY: the tasks write disjoint elements — a slot belongs to exactly one group, and
    // within a group each task owns a distinct block of output rows. `set` takes `&self` so
    // the rayon closure captures this (Send + Sync) wrapper, not the bare pointer.
    struct Out(*mut f32);
    unsafe impl Send for Out {}
    unsafe impl Sync for Out {}
    impl Out {
        #[inline]
        unsafe fn set(&self, index: usize, value: f32) {
            *self.0.add(index) = value;
        }
    }
    let out = Out(dst.as_mut_ptr());
    tasks.par_iter().for_each(|((expert, xrow), chunk, slots)| {
        let w = &blocks[*expert as usize * expert_blocks..][..expert_blocks];
        let x = &xq[*xrow * vd_row..(*xrow + 1) * vd_row];
        let r0 = chunk * ROWS_PER_TASK;
        let take = (n - r0).min(ROWS_PER_TASK);
        let rows = &w[r0 * row_blocks..(r0 + take) * row_blocks];
        for (i, row) in rows.chunks_exact(row_blocks).enumerate() {
            let orow = r0 + i;
            let value = T::vec_dot(k, row, x);
            for &slot in *slots {
                // SAFETY: see `Out`.
                unsafe { out.set(slot as usize * n + orow, value) };
            }
        }
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

    /// IQ4_XS is not a candle type: `iq4xs` builds its storage and implements `GgmlType`, and
    /// the routed gemv has to serve it like any other GGUF block type.
    #[test]
    fn routed_gemv_serves_iq4xs_experts() {
        use crate::gguf::iq4xs::{self, BLOCK_BYTES, QK_K};

        let (e, n, k, tokens, topk) = (4, 16, 256, 3, 2);
        let row_bytes = (k / QK_K) * BLOCK_BYTES;
        let w: Vec<f32> = (0..e * n * k)
            .map(|i| ((i as f32) * 0.37).sin() * 0.5)
            .collect();
        let mut packed = vec![0u8; e * n * row_bytes];
        for row in 0..e * n {
            iq4xs::quantize_iq4_xs(
                &w[row * k..(row + 1) * k],
                &mut packed[row * row_bytes..(row + 1) * row_bytes],
            )
            .unwrap();
        }
        let qt = std::sync::Arc::new(
            QTensor::new(iq4xs::cpu_storage(&packed).unwrap(), (e, n, k)).unwrap(),
        );

        let ids = Tensor::new(&[[0u32, 2], [3, 1], [2, 2]], &Device::Cpu).unwrap();
        let x: Vec<f32> = (0..tokens * k).map(|i| ((i as f32) * 0.11).cos()).collect();
        let x = Tensor::from_vec(x, (tokens, 1, k), &Device::Cpu).unwrap();
        let got = super::qtensor_indexed_moe_forward(&qt, &x, &ids).unwrap();
        assert_eq!(got.dims(), &[tokens, topk, n]);

        // Reference: dequantize the expert's rows and dot them with the raw activation. The
        // kernel quantizes the activation to Q8_K, so the tolerance is a percent, not a bit.
        let ids_v = ids.to_vec2::<u32>().unwrap();
        let x_v = x.to_vec3::<f32>().unwrap();
        for (t, row_ids) in ids_v.iter().enumerate() {
            for (j, &ex) in row_ids.iter().enumerate() {
                let ex = ex as usize;
                let mut deq = vec![0f32; n * k];
                iq4xs::dequantize_scalar(
                    &packed[ex * n * row_bytes..(ex + 1) * n * row_bytes],
                    &mut deq,
                )
                .unwrap();
                for o in 0..n {
                    let want: f32 = (0..k).map(|i| deq[o * k + i] * x_v[t][0][i]).sum();
                    let g = got.i((t, j, o)).unwrap().to_scalar::<f32>().unwrap();
                    assert!(
                        (g - want).abs() <= 2e-2 * want.abs().max(1.0),
                        "IQ4_XS slot ({t},{j}) row {o}: {g} vs {want}"
                    );
                }
            }
        }
    }
}

/// Throughput microbenchmark for the per-dtype CPU vec-dot that `--cpu-moe`
/// decode drives. Run with:
///
/// ```text
/// cargo test -p mistralrs-quant --lib -- --ignored --nocapture bench_vec_dot_throughput
/// ```
#[cfg(test)]
mod throughput_bench {
    use candle_core::quantized::{
        k_quants::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0},
        GgmlType,
    };
    use rayon::prelude::*;
    use std::time::Instant;

    use crate::gguf::cpu::ROWS_PER_TASK;
    use crate::gguf::{
        iq2s::BlockIQ2S, iq3s::BlockIQ3S, iq3xxs::BlockIQ3XXS, iq4nl::BlockIQ4NL,
        iq4xs::BlockIQ4XS, q2_0::BlockQ2_0,
    };

    fn pack<T: GgmlType>(n_rows: usize, k: usize) -> Vec<u8> {
        let row_bytes = (k / T::BLCK_SIZE) * T::DTYPE.type_size();
        let mut raw = vec![0u8; n_rows * row_bytes];
        let mut s = 12345u32;
        for b in raw.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 24) as u8;
        }
        raw
    }

    fn run<T: GgmlType + Send + Sync>(
        label: &str,
        n: usize,
        k: usize,
        experts: usize,
        iters: usize,
    ) {
        let rb = k / T::BLCK_SIZE;
        let row_bytes = rb * T::DTYPE.type_size();
        let n_rows = n * experts;
        let raw = pack::<T>(n_rows, k);
        assert_eq!(raw.as_ptr().align_offset(std::mem::align_of::<T>()), 0);
        // SAFETY: packed blocks laid out for `T`.
        let blocks: &[T] =
            unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<T>(), n_rows * rb) };

        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let vd = k / T::VecDotType::BLCK_SIZE;
        let mut xq = vec![T::VecDotType::zeros(); vd];
        T::VecDotType::from_float(&x, &mut xq);

        let bytes = (n_rows * row_bytes) as f64;
        let mut out = vec![0f32; n_rows];

        let t = Instant::now();
        for _ in 0..iters {
            for (o, row) in out.iter_mut().zip(blocks.chunks_exact(rb)) {
                *o = T::vec_dot(k, row, &xq);
            }
        }
        let s1 = t.elapsed().as_secs_f64();

        let mut dst = vec![0f32; n_rows];
        let t = Instant::now();
        for _ in 0..iters {
            dst.par_chunks_mut(n).enumerate().for_each(|(e, out)| {
                let w = &blocks[e * n * rb..][..n * rb];
                out.par_chunks_mut(ROWS_PER_TASK)
                    .zip(w.par_chunks(ROWS_PER_TASK * rb))
                    .for_each(|(out, w)| {
                        for (o, row) in out.iter_mut().zip(w.chunks_exact(rb)) {
                            *o = T::vec_dot(k, row, &xq);
                        }
                    });
            });
        }
        let sm = t.elapsed().as_secs_f64();

        println!(
            "{label:<8} {n:>5}x{k:<5} {experts:>2}e  1T {:5.2} GB/s   split {:5.2} GB/s  {:6.0} us/call",
            bytes * iters as f64 / s1 / 1e9,
            bytes * iters as f64 / sm / 1e9,
            sm / iters as f64 * 1e6
        );
    }

    #[test]
    #[ignore = "throughput microbenchmark"]
    fn bench_vec_dot_throughput() {
        println!("rayon threads: {}", rayon::current_num_threads());
        // Working-set sweep: 10 experts (~19 MB) fits in the 32 MB L3 of a 3600X and
        // measures the kernel ceiling; 128 (~240 MB) forces DRAM and shows how much
        // the single memory channel costs each type.
        let e: usize = std::env::var("EXPERTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let iters = (200 / e).max(2);
        println!("--- gate/up (n=640, k=2560), {e} experts/token, {iters} iters ---");
        let (n, k) = (640usize, 2560usize);
        run::<BlockIQ2S>("IQ2_S", n, k, e, iters);
        run::<BlockIQ3XXS>("IQ3_XXS", n, k, e, iters);
        run::<BlockIQ3S>("IQ3_S", n, k, e, iters);
        run::<BlockIQ4NL>("IQ4_NL", n, k, e, iters);
        run::<BlockIQ4XS>("IQ4_XS", n, k, e, iters);
        println!("--- candle K-quants (requant baseline) ---");
        run::<BlockQ2K>("Q2_K", n, k, e, iters);
        run::<BlockQ3K>("Q3_K", n, k, e, iters);
        run::<BlockQ4K>("Q4_K", n, k, e, iters);
        println!("--- down (n=2560, k=640), {e} experts/token ---");
        let (n, k) = (2560usize, 640usize);
        run::<BlockQ2_0>("Q2_0", n, k, e, iters);
        run::<BlockIQ4NL>("IQ4_NL", n, k, e, iters);
        run::<BlockQ4_0>("Q4_0", n, k, e, iters);
    }
}
