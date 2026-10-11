//! CPU dequantization of the GGML types this crate implements.
//!
//! candle stores our IQ blocks but cannot dequantize them on CUDA: the block
//! types live here, not in candle, so `QCudaStorage::dequantize` and the
//! quantized-embedding `get_rows` kernels bail for them. The routed-expert and
//! matmul paths use this crate's own CUDA kernels and never need either, but a
//! quantized *token embedding* does -- so `GgufMatMul` dequantizes the requested
//! rows here, on the host, and copies the result back.

use candle_core::{quantized::GgmlDType, Result};

use super::{iq2s, iq3s, iq3xxs, iq4nl, iq4xs, q2_0};

/// True for the GGML types this crate implements (see `IQ_FORMATS.md`).
pub fn is_fork_iq(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::IQ4XS
            | GgmlDType::IQ4NL
            | GgmlDType::Q2_0
            | GgmlDType::IQ3_XXS
            | GgmlDType::IQ2_S
            | GgmlDType::IQ3_S
    )
}

/// Dequantize `bytes` (`dst.len()` packed blocks) into `dst`.
pub fn dequantize(bytes: &[u8], dtype: GgmlDType, dst: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlDType::IQ4XS => iq4xs::dequantize(bytes, dst),
        GgmlDType::IQ4NL => iq4nl::dequantize(bytes, dst),
        GgmlDType::Q2_0 => q2_0::dequantize(bytes, dst),
        GgmlDType::IQ3_XXS => iq3xxs::dequantize(bytes, dst),
        GgmlDType::IQ2_S => iq2s::dequantize(bytes, dst),
        GgmlDType::IQ3_S => iq3s::dequantize(bytes, dst),
        other => candle_core::bail!("no CPU dequantizer for {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fork_iq_covers_exactly_the_implemented_formats() {
        for dtype in [
            GgmlDType::IQ4XS,
            GgmlDType::IQ4NL,
            GgmlDType::Q2_0,
            GgmlDType::IQ3_XXS,
            GgmlDType::IQ2_S,
            GgmlDType::IQ3_S,
        ] {
            assert!(is_fork_iq(dtype), "{dtype:?} should be handled here");
        }
        for dtype in [
            GgmlDType::F16,
            GgmlDType::Q4K,
            GgmlDType::Q8_0,
            GgmlDType::Q8K,
        ] {
            assert!(!is_fork_iq(dtype), "{dtype:?} is candle's, not ours");
        }
    }

    #[test]
    fn dequantize_dispatches_by_dtype() {
        // Q2_0: 64 elements per 18-byte block.
        let mut q2_0 = [0f32; 64];
        dequantize(&[0u8; 18], GgmlDType::Q2_0, &mut q2_0).unwrap();
        // IQ2_S: 256 elements per 82-byte block.
        let mut iq2s = [0f32; 256];
        dequantize(&[0u8; 82], GgmlDType::IQ2_S, &mut iq2s).unwrap();
        // A dtype candle owns has no kernel here.
        let mut other = [0f32; 256];
        assert!(dequantize(&[0u8; 144], GgmlDType::Q4K, &mut other).is_err());
        // A length that does not match the block count is rejected.
        assert!(dequantize(&[0u8; 17], GgmlDType::Q2_0, &mut q2_0).is_err());
    }
}
