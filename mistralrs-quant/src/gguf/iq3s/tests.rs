//! Unit tests for the IQ3_S CPU kernels.
//!
//! Dequantization and the vec-dot are checked against ggml
//! (`tools/iq3s/oracle.cpp`). The quantizer is a plain grid search, so only its
//! round-trip error is tested.

use super::*;

const BLOB: &[u8] = include_bytes!("../testdata/iq3_s_oracle.bin");

fn u32at(o: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([o[i], o[i + 1], o[i + 2], o[i + 3]])
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

struct Oracle {
    nb_real: usize,
    nb_syn: usize,
    real_raw: &'static [u8],
    real_deq: Vec<f32>,
    real_act: Vec<f32>,
    real_q8: &'static [u8],
    real_dot: Vec<f32>,
    syn_raw: &'static [u8],
    syn_deq: Vec<f32>,
    syn_act: Vec<f32>,
    syn_q8: &'static [u8],
    syn_dot: Vec<f32>,
}

fn oracle() -> Oracle {
    assert_eq!(&BLOB[..8], b"IQ3SOR01", "oracle blob magic");
    let nb_real = u32at(BLOB, 8) as usize;
    let nb_syn = u32at(BLOB, 12) as usize;
    assert!(nb_real > 0 && nb_syn > 0);
    let mut o = 24usize;
    let real_raw = &BLOB[o..o + nb_real * BLOCK_BYTES];
    o += nb_real * BLOCK_BYTES;
    let real_deq = f32s(&BLOB[o..o + nb_real * QK_K * 4]);
    o += nb_real * QK_K * 4;
    let real_act = f32s(&BLOB[o..o + nb_real * QK_K * 4]);
    o += nb_real * QK_K * 4;
    let real_q8 = &BLOB[o..o + nb_real * Q8K_BYTES];
    o += nb_real * Q8K_BYTES;
    let real_dot = f32s(&BLOB[o..o + nb_real * 4]);
    o += nb_real * 4;
    let syn_raw = &BLOB[o..o + nb_syn * BLOCK_BYTES];
    o += nb_syn * BLOCK_BYTES;
    let syn_deq = f32s(&BLOB[o..o + nb_syn * QK_K * 4]);
    o += nb_syn * QK_K * 4;
    let syn_act = f32s(&BLOB[o..o + nb_syn * QK_K * 4]);
    o += nb_syn * QK_K * 4;
    let syn_q8 = &BLOB[o..o + nb_syn * Q8K_BYTES];
    o += nb_syn * Q8K_BYTES;
    let syn_dot = f32s(&BLOB[o..o + nb_syn * 4]);
    o += nb_syn * 4;
    assert_eq!(o, BLOB.len(), "oracle blob has trailing bytes");
    Oracle {
        nb_real,
        nb_syn,
        real_raw,
        real_deq,
        real_act,
        real_q8,
        real_dot,
        syn_raw,
        syn_deq,
        syn_act,
        syn_q8,
        syn_dot,
    }
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: block {} element {} got {g} want {w}",
            i / QK_K,
            i % QK_K
        );
    }
}

struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Rng(seed)
    }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn activations(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

#[test]
fn constants_are_consistent_with_ggml() {
    assert_eq!(QK_K, 256);
    assert_eq!(BLOCK_BYTES, 2 + 64 + 8 + 32 + 4);
    assert_eq!(Q8K_BYTES, 292);
    assert_eq!(IQ3S_GRID.len(), 512);
}

#[test]
fn dequantize_matches_ggml() {
    let o = oracle();
    for (raw, want, nb, what) in [
        (o.real_raw, &o.real_deq, o.nb_real, "real"),
        (o.syn_raw, &o.syn_deq, o.nb_syn, "syn"),
    ] {
        let mut got = vec![0f32; nb * QK_K];
        dequantize(raw, &mut got).unwrap();
        assert_bits_eq(&got, want, what);
    }
}

#[test]
fn q8_k_quantize_matches_ggml() {
    let o = oracle();
    for (act, want, nb, what) in [
        (&o.real_act, o.real_q8, o.nb_real, "real"),
        (&o.syn_act, o.syn_q8, o.nb_syn, "syn"),
    ] {
        let mut got = vec![0u8; nb * Q8K_BYTES];
        quantize_row_q8_k(act, &mut got).unwrap();
        assert_eq!(got.as_slice(), want, "{what} q8_K");
    }
}

#[test]
fn vec_dot_matches_reference() {
    let o = oracle();
    for (raw, q8, dots, nb, what) in [
        (o.real_raw, o.real_q8, &o.real_dot, o.nb_real, "real"),
        (o.syn_raw, o.syn_q8, &o.syn_dot, o.nb_syn, "syn"),
    ] {
        for b in 0..nb {
            let got = vec_dot_q8_k(
                &raw[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
                &q8[b * Q8K_BYTES..(b + 1) * Q8K_BYTES],
            )
            .unwrap();
            assert!(
                close(got, dots[b], 1e-4),
                "{what} block {b}: {got} vs {}",
                dots[b]
            );
        }
    }
}

#[test]
fn quantize_round_trip_error_is_bounded() {
    let mut rng = Rng::new(0x0bad_f00d);
    let x = rng.activations(4 * QK_K);
    let mut packed = vec![0u8; 4 * BLOCK_BYTES];
    quantize_iq3_s(&x, &mut packed).unwrap();
    let mut back = vec![0f32; 4 * QK_K];
    dequantize(&packed, &mut back).unwrap();
    let mut sig = 0f64;
    let mut err = 0f64;
    for (a, b) in x.iter().zip(&back) {
        sig += (*a as f64) * (*a as f64);
        let d = (*a as f64) - (*b as f64);
        err += d * d;
    }
    let rel = (err / sig).sqrt();
    assert!(rel < 0.4, "relative RMSE {rel} is too large");
}

#[test]
fn const_dtype_and_block_size() {
    assert_eq!(BlockIQ3S::DTYPE, GgmlDType::IQ3_S);
    assert_eq!(BlockIQ3S::BLCK_SIZE, QK_K);
}

#[test]
fn block_type_matches_the_byte_kernel() {
    let o = oracle();
    let blocks = blocks_from_bytes(o.real_raw).unwrap();
    let mut got = vec![0f32; o.nb_real * QK_K];
    BlockIQ3S::to_float(&blocks, &mut got);
    assert_bits_eq(&got, &o.real_deq, "GgmlType::to_float");

    let mut q8 = vec![BlockQ8K::zeros(); o.nb_real];
    // SAFETY: `BlockQ8K` is repr(C) and unpadded (candle asserts its size).
    unsafe { as_bytes_mut(&mut q8) }.copy_from_slice(o.real_q8);
    for b in 0..o.nb_real {
        let got = BlockIQ3S::vec_dot(QK_K, &blocks[b..b + 1], &q8[b..b + 1]);
        let w = &o.real_raw[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let q = &o.real_q8[b * Q8K_BYTES..(b + 1) * Q8K_BYTES];
        let want = vec_dot_q8_k(w, q).unwrap();
        assert!(close(got, want, 1e-6), "block {b}: {got} vs {want}");
    }
}

#[test]
fn cpu_storage_builds_an_iq3s_qtensor() {
    use candle_core::quantized::QTensor;
    let o = oracle();
    let qt = QTensor::new(cpu_storage(o.real_raw).unwrap(), vec![o.nb_real * QK_K]).unwrap();
    assert_eq!(qt.dtype(), GgmlDType::IQ3_S);
    assert_eq!(&*qt.data().unwrap(), o.real_raw);
}

#[test]
fn gemv_matches_a_dequantized_reference() {
    let o = oracle();
    let x = &o.real_act[..QK_K];
    let rows = o.nb_real;
    let mut out = vec![0f32; rows];
    gemv(o.real_raw, x, &mut out).unwrap();

    let mut q8 = vec![0u8; Q8K_BYTES];
    quantize_row_q8_k(x, &mut q8).unwrap();
    let d = rd_f32(&q8, 0);
    let act: Vec<f32> = (0..QK_K).map(|i| d * q8[4 + i] as i8 as f32).collect();
    for (r, &got) in out.iter().enumerate() {
        let w = &o.real_deq[r * QK_K..(r + 1) * QK_K];
        let want: f64 = w
            .iter()
            .zip(&act)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        assert!(close(got, want as f32, 1e-4), "row {r}: {got} vs {want}");
    }
}

#[test]
fn bad_lengths_are_rejected() {
    let mut out = [0f32; QK_K];
    assert!(dequantize(&[0u8; BLOCK_BYTES - 1], &mut out).is_err());
    assert!(dequantize(&[0u8; BLOCK_BYTES], &mut out[..QK_K - 1]).is_err());
    assert!(quantize_row_q8_k(&[0f32; QK_K - 1], &mut [0u8; Q8K_BYTES]).is_err());
    assert!(quantize_iq3_s(&[0f32; QK_K - 1], &mut [0u8; BLOCK_BYTES]).is_err());
    assert!(quantize_iq3_s(&[0f32; QK_K], &mut [0u8; BLOCK_BYTES - 1]).is_err());
    assert!(vec_dot_q8_k(&[0u8; BLOCK_BYTES - 1], &[0u8; Q8K_BYTES]).is_err());
    assert!(vec_dot_q8_k(&[0u8; BLOCK_BYTES], &[0u8; Q8K_BYTES - 1]).is_err());
    assert!(gemv(&[0u8; BLOCK_BYTES], &[0f32; QK_K - 4], &mut [0f32; 1]).is_err());
    assert!(gemv(&[0u8; BLOCK_BYTES], &[0f32; QK_K], &mut [0f32; 2]).is_err());
}

// ---------------------------------------------------------------------------
// CUDA: the same kernels as the other formats, checked against the CPU kernel
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn pack_rows(f: impl Fn(usize) -> f32, rows: usize, k: usize) -> Vec<u8> {
    let row_bytes = (k / QK_K) * BLOCK_BYTES;
    let mut out = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        let row: Vec<f32> = (0..k).map(|i| f(i + r * k)).collect();
        quantize_iq3_s(&row, &mut out[r * row_bytes..(r + 1) * row_bytes]).unwrap();
    }
    out
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_indexed_moe_matches_the_cpu_kernel() {
    use candle_core::{DType, Device, Tensor};
    use std::sync::Arc;
    let (e, n, k, tokens, topk) = (4, 16, 256, 3, 2);
    let row_bytes = (k / QK_K) * BLOCK_BYTES;
    let w: Vec<f32> = (0..e * n * k)
        .map(|i| ((i as f32) * 0.37).sin() * 0.5)
        .collect();
    let mut packed = vec![0u8; e * n * row_bytes];
    for row in 0..e * n {
        quantize_iq3_s(
            &w[row * k..(row + 1) * k],
            &mut packed[row * row_bytes..(row + 1) * row_bytes],
        )
        .unwrap();
    }
    let x: Vec<f32> = (0..tokens * k).map(|i| ((i as f32) * 0.11).cos()).collect();
    let ids_data = [[0u32, 2], [3, 1], [2, 2]];
    let mut want = vec![0f32; tokens * topk * n];
    for (t, row_ids) in ids_data.iter().enumerate() {
        for (j, &ex) in row_ids.iter().enumerate() {
            let ex = ex as usize;
            let mut out = vec![0f32; n];
            gemv(
                &packed[ex * n * row_bytes..(ex + 1) * n * row_bytes],
                &x[t * k..(t + 1) * k],
                &mut out,
            )
            .unwrap();
            want[(t * topk + j) * n..(t * topk + j + 1) * n].copy_from_slice(&out);
        }
    }
    let dev = Device::new_cuda(0).unwrap();
    let qt = Arc::new(
        crate::gguf::archive::qtensor_from_gguf_data(
            GgmlDType::IQ3_S,
            &packed,
            vec![e, n, k],
            &dev,
        )
        .unwrap(),
    );
    assert_eq!(qt.dtype(), GgmlDType::IQ3_S);
    let xd = Tensor::from_vec(x, (tokens, 1, k), &dev).unwrap();
    let ids = Tensor::new(&ids_data, &dev).unwrap();
    let got = crate::gguf::cuda::qtensor_indexed_moe_forward(&qt, &xd, &ids)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 5e-2 * w.abs().max(1.0),
            "indexed element {i}: {g} vs {w}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mmvq_matches_the_cpu_kernel() {
    use candle_core::{DType, Device, Tensor};
    let (n, k) = (32, 256);
    let packed = pack_rows(|i| ((i as f32) * 0.29).sin().abs() * 0.5 + 0.1, n, k);
    let x: Vec<f32> = (0..k)
        .map(|i| ((i as f32) * 0.17).cos().abs() + 0.5)
        .collect();
    let mut want = vec![0f32; n];
    gemv(&packed, &x, &mut want).unwrap();
    let dev = Device::new_cuda(0).unwrap();
    let qt =
        crate::gguf::archive::qtensor_from_gguf_data(GgmlDType::IQ3_S, &packed, vec![n, k], &dev)
            .unwrap();
    let xd = Tensor::from_vec(x, (1, k), &dev).unwrap();
    let got = crate::gguf::fast_mmvq::plain(&qt, &xd)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 5e-2 * w.abs().max(1.0),
            "mmvq row {i}: {g} vs {w}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mmvq_chunked_handles_batch_over_eight() {
    use candle_core::{DType, Device, Tensor};
    let (n, k, batch) = (32, 256, 16);
    let packed = pack_rows(|i| ((i as f32) * 0.29).sin().abs() * 0.5 + 0.1, n, k);
    let x: Vec<f32> = (0..batch * k)
        .map(|i| ((i as f32) * 0.07).cos().abs() + 0.5)
        .collect();
    let mut want = vec![0f32; batch * n];
    for t in 0..batch {
        gemv(
            &packed,
            &x[t * k..(t + 1) * k],
            &mut want[t * n..(t + 1) * n],
        )
        .unwrap();
    }
    let dev = Device::new_cuda(0).unwrap();
    let qt =
        crate::gguf::archive::qtensor_from_gguf_data(GgmlDType::IQ3_S, &packed, vec![n, k], &dev)
            .unwrap();
    let xd = Tensor::from_vec(x, (batch, k), &dev).unwrap();
    let got = crate::gguf::GgufMatMul::mmvq_chunked(&qt, &xd)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 5e-2 * w.abs().max(1.0),
            "chunked element {i}: {g} vs {w}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_indexed_moe_fused_decode_matches_the_cpu_kernel() {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Device, Storage, Tensor};
    const HIDDEN: usize = 256;
    const INTER: usize = 256;
    const E: usize = 4;
    const BATCH: usize = 3;
    const TOPK: usize = 2;
    let gate = pack_rows(
        |i| ((i as f32) * 0.13).sin().abs() * 0.4 + 0.1,
        E * INTER,
        HIDDEN,
    );
    let up = pack_rows(
        |i| ((i as f32) * 0.17).cos().abs() * 0.4 + 0.1,
        E * INTER,
        HIDDEN,
    );
    let down = pack_rows(
        |i| ((i as f32) * 0.19).sin().abs() * 0.4 + 0.1,
        E * HIDDEN,
        INTER,
    );
    let rg = (HIDDEN / QK_K) * BLOCK_BYTES;
    let rd = (INTER / QK_K) * BLOCK_BYTES;
    let x: Vec<f32> = (0..BATCH * HIDDEN)
        .map(|i| ((i as f32) * 0.11).cos().abs() + 0.5)
        .collect();
    let ids_data: [[u32; 2]; 3] = [[0, 2], [3, 1], [2, 2]];
    let weights: [[f32; 2]; 3] = [[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]];
    let mut want = vec![0f32; BATCH * HIDDEN];
    for t in 0..BATCH {
        let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
        for j in 0..TOPK {
            let ex = ids_data[t][j] as usize;
            let mut g = vec![0f32; INTER];
            gemv(&gate[ex * INTER * rg..(ex + 1) * INTER * rg], xt, &mut g).unwrap();
            let mut u = vec![0f32; INTER];
            gemv(&up[ex * INTER * rg..(ex + 1) * INTER * rg], xt, &mut u).unwrap();
            let act: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let mut d = vec![0f32; HIDDEN];
            gemv(
                &down[ex * HIDDEN * rd..(ex + 1) * HIDDEN * rd],
                &act,
                &mut d,
            )
            .unwrap();
            let w = weights[t][j];
            for o in 0..HIDDEN {
                want[t * HIDDEN + o] += w * d[o];
            }
        }
    }
    let dev = Device::new_cuda(0).unwrap();
    let cuda = dev.as_cuda_device().unwrap();
    let gate_qt = crate::gguf::archive::qtensor_from_gguf_data(
        GgmlDType::IQ3_S,
        &gate,
        vec![E, INTER, HIDDEN],
        &dev,
    )
    .unwrap();
    let up_qt = crate::gguf::archive::qtensor_from_gguf_data(
        GgmlDType::IQ3_S,
        &up,
        vec![E, INTER, HIDDEN],
        &dev,
    )
    .unwrap();
    let down_qt = crate::gguf::archive::qtensor_from_gguf_data(
        GgmlDType::IQ3_S,
        &down,
        vec![E, HIDDEN, INTER],
        &dev,
    )
    .unwrap();
    let xs = Tensor::from_vec(x, (BATCH, HIDDEN), &dev).unwrap();
    let ids = Tensor::new(&ids_data, &dev)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap();
    let (ids_storage, _) = ids.storage_and_layout();
    let Storage::Cuda(ids_cuda) = &*ids_storage else {
        panic!("ids must be on CUDA")
    };
    let ids_slice = ids_cuda.as_cuda_slice::<u32>().unwrap();
    let wt = Tensor::new(&weights, &dev)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let (wt_storage, _) = wt.storage_and_layout();
    let Storage::Cuda(wt_cuda) = &*wt_storage else {
        panic!("weights must be on CUDA")
    };
    let wt_slice = wt_cuda.as_cuda_slice::<f32>().unwrap();
    let wt_ptr = wt_slice.device_ptr(wt_slice.stream()).0 as *const f32;
    // SAFETY: contiguous, on this device; `wt_ptr` points at BATCH*TOPK f32.
    let got = unsafe {
        crate::gguf::cuda::indexed_moe_fused_decode(
            &gate_qt,
            &up_qt,
            &down_qt,
            &xs,
            ids_slice,
            wt_ptr,
            BATCH,
            TOPK,
            crate::gguf::cuda::ACT_SILU,
            cuda,
        )
    }
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    let mut max_rel = 0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let rel = (g - w).abs() / w.abs().max(1.0);
        max_rel = max_rel.max(rel);
        assert!(rel <= 0.1, "fused decode element {i}: {g} vs {w}");
    }
    eprintln!("IQ3_S fused decode max relative error: {max_rel:.4}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_grouped_moe_gemm_matches_the_cpu_kernel() {
    use candle_core::{DType, Device, Storage, Tensor};
    let (e, n, k) = (4usize, 32usize, 256usize);
    let (tokens, topk) = (6usize, 2usize);
    let total = tokens * topk;
    let row_bytes = (k / QK_K) * BLOCK_BYTES;
    let w: Vec<f32> = (0..e * n * k)
        .map(|i| ((i as f32) * 0.23).sin().abs() * 0.5 + 0.1)
        .collect();
    let mut packed = vec![0u8; e * n * row_bytes];
    for row in 0..e * n {
        quantize_iq3_s(
            &w[row * k..(row + 1) * k],
            &mut packed[row * row_bytes..(row + 1) * row_bytes],
        )
        .unwrap();
    }
    let x: Vec<f32> = (0..tokens * k)
        .map(|i| ((i as f32) * 0.07).cos().abs() + 0.5)
        .collect();
    let ids_data: [[u32; 2]; 6] = [[0, 2], [1, 3], [2, 0], [3, 1], [0, 3], [2, 2]];
    let dev = Device::new_cuda(0).unwrap();
    let cuda = dev.as_cuda_device().unwrap();
    let qt = crate::gguf::archive::qtensor_from_gguf_data(
        GgmlDType::IQ3_S,
        &packed,
        vec![e, n, k],
        &dev,
    )
    .unwrap();
    let xd = Tensor::from_vec(x.clone(), (tokens, k), &dev).unwrap();
    let (input_q8, qk, qk_padded) = crate::gguf::cuda::quantize_input_q8_1(&xd, cuda).unwrap();
    let ids = Tensor::new(&ids_data, &dev)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap();
    let (ids_storage, _) = ids.storage_and_layout();
    let Storage::Cuda(ids_cuda) = &*ids_storage else {
        panic!("ids must be on CUDA")
    };
    let ids_slice = ids_cuda.as_cuda_slice::<u32>().unwrap();
    let (bounds, sorted_ids, _) =
        crate::gguf::cuda::moe_dispatch_build(ids_slice, total, e, topk, cuda).unwrap();
    let bounds_v = cuda.clone_dtoh(&bounds).unwrap();
    let sorted_v = cuda.clone_dtoh(&sorted_ids).unwrap();
    let out = crate::gguf::cuda::grouped_moe_gemm_prequantized(
        &qt,
        &input_q8,
        qk,
        qk_padded,
        &bounds,
        &sorted_ids,
        None,
        total,
        topk,
        e,
        1,
        cuda,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    let mut want = vec![0f32; total * n];
    for ti in 0..total {
        let expert = (0..e)
            .find(|&ee| (bounds_v[ee] as usize) <= ti && ti < bounds_v[ee + 1] as usize)
            .unwrap();
        let token = sorted_v[ti] as usize / topk;
        gemv(
            &packed[expert * n * row_bytes..(expert + 1) * n * row_bytes],
            &x[token * k..(token + 1) * k],
            &mut want[ti * n..(ti + 1) * n],
        )
        .unwrap();
    }
    for (i, (g, w)) in out.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 5e-2 * w.abs().max(1.0),
            "grouped element {i}: {g} vs {w}"
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_and_scalar_agree() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let o = oracle();
    for (raw, q8, nb, what) in [
        (o.real_raw, o.real_q8, o.nb_real, "real"),
        (o.syn_raw, o.syn_q8, o.nb_syn, "syn"),
    ] {
        for b in 0..nb {
            let w = &raw[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let q = &q8[b * Q8K_BYTES..(b + 1) * Q8K_BYTES];
            let s = vec_dot_q8_k_scalar(w, q).unwrap();
            // SAFETY: guarded by the runtime AVX2 check.
            let a = unsafe { vec_dot_q8_k_avx2(w, q) }.unwrap();
            assert!(
                (s - a).abs() <= 1e-4 * s.abs().max(1.0),
                "{what} block {b}: scalar {s} vs avx2 {a}"
            );
        }
    }
}
