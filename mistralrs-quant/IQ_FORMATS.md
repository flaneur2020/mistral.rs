# IQ quant formats

Status of native GGML i-quant support in `mistralrs-quant`, so the published
Qwen3.8-Flash-Next GGUFs load as published instead of being re-quantized offline.

The block type, the kernels and the tests live here. `candle-core` only carries
the `GgmlDType` variant (see below); the loader builds the storage with
`iq*::storage_for_device` (CPU: `QStorage::Cpu`, CUDA: `cuda::load_quantized`).

## Done

| Format | GGML id | Block | Paired activation | CPU | CUDA |
| --- | ---: | --- | --- | --- | --- |
| IQ4_XS | 23 | 256 el / 136 B | Q8_K | scalar + AVX2 | `indexed_moe`, `mmvq_gguf`, `moe_grouped`, `mmq_gguf` |
| IQ4_NL | 20 | 32 el / 18 B | Q8_0 | scalar + AVX2 | `indexed_moe`, `mmvq_gguf`, `moe_grouped`, `mmq_gguf` |
| Q2_0 | 42 | 64 el / 18 B | Q8_0 (2 per block) | scalar | `indexed_moe`, `mmvq_gguf`, `moe_grouped` |
| IQ3_XXS | 18 | 256 el / 98 B | Q8_K | scalar | `indexed_moe`, `mmvq_gguf`, `moe_grouped` |

Each has: an oracle generated from ggml (`tools/iq4xs`, `tools/iq4nl`,
`tools/q2_0`, `tools/iq3xxs`), CPU oracle tests, and CUDA tests that compare
every kernel against the CPU kernel. Q2_0 has no MMQ kernel (dense batches over
8 chunk through MMVQ) and no SIMD dot (ggml has none either); IQ3_XXS has no MMQ
kernel either (the fork's `mmq_common.cuh` ships a zeroed grid stub). IQ3_XXS's
*quantizer* is a plain grid search, not ggml's map/neighbour search, so only its
round-trip is tested -- the read path (dequantize + vec-dot) is checked against
ggml.

## Remaining for the Coder GGUF

From a census of shard 1 (1223 tensors); the routed experts use a mix:

| Format | GGML id | Tensors (shard 1) | Priority |
| --- | ---: | ---: | --- |
| IQ2_S | 22 | 40 exps | next (most experts); `iq2s_grid[1024]` |
| IQ3_S | 21 | 20 exps | `iq3s_grid[512]` + separate signs |

The PLE n-gram table is IQ4_NL and is read directly by the fork, so it needs no
candle support.

## Other formats Strata carries

Not needed by this model but present in Strata / other pack sizes: IQ1_S (19),
IQ1_M (29), IQ2_XXS (16), IQ2_XS (17), TQ1_0/TQ2_0 (34/35), MXFP4 (39),
NVFP4 (40). Each is the same shape of work as below.

## Recipe for one more format

1. **CPU** — `src/gguf/iq<name>.rs`: `Block` type + `impl GgmlType`
   (256-element types pair with `BlockQ8_K`, 32-element with `BlockQ8_0`), the
   scalar + AVX2 dequantize, the reference quantizer, the vec-dot and a `gemv`.
   Port from ggml's `ggml-quants.c` / `ggml-cpu`.
2. **Oracle** — `tools/iq<name>/{oracle.cpp,extract_real.py,README.md}` linked
   against `~/requant/obj`, plus `src/gguf/testdata/iq<name>_oracle.bin`. Point
   the quantizer at the model's own function (e.g. `quantize_iq4_nl`, `ntry = 7`),
   not the `_ref` wrapper when they differ.
3. **candle** — in the local `../candle` fork: add `GgmlDType::IQ<Name>` (enum,
   `from_u32`/`to_u32`, `block_size`, `type_size`) and the stub arms in
   `cpu_zeros`, `from_data`, `QStorage::from_data` (Metal + CUDA) and
   `QCudaStorage::dequantize` (the block type lives here, not in candle). ~20 lines.
4. **Fork wiring** — `archive.rs` (`candle_dtype`, `ggml_dtype_alignment`,
   `storage_for_device` dispatch), `cpu.rs` (`routed_gemv` arm), `gguf/mod.rs`
   (`mod`, uqff code/label), `lib.rs` (`TryFrom<GgmlDType> for IsqType` bail arm),
   `packed_affine.rs` (return `None`).
5. **CUDA** — `indexed_moe.cu` (block, `vec_dot`, indexed + fused gate/up +
   down/aggregate + launchers), `mmvq_gguf.cu` (block, `vec_dot` with `kbx`,
   plain/glu/qkv batch sets + launchers), `moe_grouped.cu` (block, `vd_` wrapper,
   `MOE_KERNEL_INST` + `LAUNCH_MOE_TILED_GEMM`), and
   `mmq_gguf/mmq_instance_iq<name>.cu` (copy an instance; the type traits, tiles
   and `vec_dot` often already ship in `mmq_gguf.cuh`). Then `ffi.rs`,
   `cuda.rs` (`indexed_moe_weight_dtype`, `indexed_moe_forward_fused_q8_1_input`,
   `indexed_moe_fused_decode`, `grouped_moe_gemm_prequantized`,
   `supports_moe_grouped`), `fast_mmvq.rs`, `fast_mmq.rs`.
   IQ3/IQ2 need ggml's codebook grid tables (`ggml-common.h`), which are large.
6. **Tests** — CPU oracle tests plus CUDA tests against the CPU kernel (indexed
   MoE, dense MMVQ, dense MMQ, chunked MMVQ, fused decode, grouped prefill).

## Notes

- A `block_iq4_nl`'s `qs` sits at byte offset 2, so the CUDA code must use the
  2-byte-safe `get_int_from_uint8`, not the 4-byte `..._aligned` variant (the
  latter faults with `CUDA_ERROR_MISALIGNED_ADDRESS`). `block_iq4_xs::qs` is at
  offset 8 and is fine either way; `block_q2_0::qs` is read byte-wise.
- Q2_0 has no MMQ path in the fork's `mmq_gguf` headers, so `fast_mmq` must not
  claim it (`supports(Q2_0)` stays false); the dense prompt path uses
  `GgufMatMul::mmvq_chunked`. The same holds for IQ3_XXS: `mmq_common.cuh`
  stubs `iq3xxs_grid` (and the other iq grids) to zero, so add the real table
  plus a `mmq_instance_iq3_xxs.cu` before claiming it.
- `supports_mmq(dtype)` being true makes the MoE prefill use the MMQ pair path;
  `grouped_moe_gemm_prequantized` (`moe_grouped.cu`) is the fallback and is also
  wired and tested.
- A dense batch over 8 without an MMQ kernel chunks through MMVQ
  (`GgufMatMul::mmvq_chunked`).
- To use this from `naiwa-rs`, push the fork and the candle fork and move the
  pinned `mistralrs` rev; `Cargo.toml` here points candle at `../candle`.
