# IQ quant formats

Status of native GGML i-quant support in `mistralrs-quant`, so the published
Qwen3.8-Flash-Next GGUFs load as published instead of being re-quantized offline.

The block type, the kernels and the tests live here. `candle-core` only carries
the `GgmlDType` variant (see below); the loader builds the storage with
`iq*::storage_for_device` (CPU: `QStorage::Cpu`, CUDA: `cuda::load_quantized`).

## Done

| Format | GGML id | Block | Paired activation | CPU | CUDA |
| --- | ---: | --- | --- | --- | --- |
| IQ4_XS | 23 | 256 el / 136 B | Q8_K | scalar + AVX2 | `indexed_moe`, `mmvq`, `moe_grouped`, `mmq` |
| IQ4_NL | 20 | 32 el / 18 B | Q8_0 | scalar + AVX2 | `indexed_moe`, `mmvq`, `moe_grouped`, `mmq` |
| Q2_0 | 42 | 64 el / 18 B | Q8_0 (2 per block) | scalar | `indexed_moe`, `mmvq`, `moe_grouped`, `mmq` |
| IQ3_XXS | 18 | 256 el / 98 B | Q8_K | scalar + AVX2 | `indexed_moe`, `mmvq`, `moe_grouped`, `mmq` |
| IQ2_S | 22 | 256 el / 82 B | Q8_K | scalar + AVX2 | `indexed_moe`, `mmvq`, `moe_grouped`, `mmq` |
| IQ3_S | 21 | 256 el / 110 B | Q8_K | scalar + AVX2 | `indexed_moe`, `mmvq`, `moe_grouped`, `mmq` |

Every format now covers the whole CUDA surface: the per-token `indexed_moe`
kernel, dense MMVQ (batch 1-8), the chunked MMVQ fallback for empty rows, the
fused decode (`indexed_moe_fused_decode`) and the grouped prefill (both the
`moe_grouped` kernel and the `mmq` pair path).

Each has: an oracle generated from ggml (`tools/iq4xs`, `tools/iq4nl`,
`tools/q2_0`, `tools/iq3xxs`, `tools/iq2s`, `tools/iq3_s`), CPU oracle tests, and
CUDA tests that compare every kernel against the CPU kernel (indexed MoE, dense
MMVQ, chunked MMVQ, fused decode, grouped prefill, dense MMQ, grouped MMQ), plus
an AVX2-vs-scalar agreement test where a SIMD dot exists.

Only Q2_0 has a scalar-only CPU dot: ggml itself ships no x86 SIMD for it
(arm/loongarch/wasm only), so there is nothing to port. The IQ3_XXS/IQ2_S/IQ3_S
*quantizers* are plain grid searches, not ggml's map/neighbour search, so only
their round-trip is tested -- the read path (dequantize + vec-dot) is checked
against ggml.

## The Coder GGUF

The published Qwen3.8-Flash-Next shards use IQ2_S, IQ3_XXS, IQ3_S, Q2_0, IQ4_NL
and IQ4_XS on the routed experts (plus IQ4_NL for the PLE n-gram table, which the
fork reads directly), so the model now loads as published -- no offline
re-quantisation. A census of shard 1 (1223 tensors) found:

| Format | GGML id | Tensors (shard 1) |
| --- | ---: | ---: |
| IQ4_NL | 20 | 86 (39 exps + 47 other) |
| IQ4_XS | 23 | 45 |
| IQ2_S | 22 | 40 exps |
| IQ3_XXS | 18 | 34 |
| IQ3_S | 21 | 20 exps |
| Q2_0 | 42 | 9 |

## Other formats Strata carries

Not needed by this model but present in Strata / other pack sizes: IQ1_S (19),
IQ1_M (29), IQ2_XXS (16), IQ2_XS (17), TQ1_0/TQ2_0 (34/35), MXFP4 (39),
NVFP4 (40). Each is the same shape of work as below.

## Recipe for one more format

1. **CPU** — `src/gguf/iq<name>.rs`: `Block` type + `impl GgmlType`
   (256-element types pair with `BlockQ8_K`, 32-element with `BlockQ8_0`), the
   scalar dequantize, the reference quantizer, the scalar vec-dot and a `gemv`.
   Port from ggml's `ggml-quants.c` / `ggml-cpu`. If ggml ships an x86 SIMD dot
   (`ggml-cpu/arch/x86/quants.c`), port its `__AVX2__` branch too, dispatch on
   `is_x86_feature_detected!("avx2")` and add an `avx2_and_scalar_agree` test.
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
5. **CUDA MMVQ/MoE** — `indexed_moe.cu` (block, `vec_dot`, indexed + fused
   gate/up + down/aggregate + launchers), `mmvq_gguf.cu` (block, `vec_dot` with
   `kbx`, plain/glu/qkv batch sets + launchers), `moe_grouped.cu` (block, `vd_`
   wrapper, `MOE_KERNEL_INST` + `LAUNCH_MOE_TILED_GEMM`). Then `ffi.rs`,
   `cuda.rs` (`indexed_moe_weight_dtype`, `indexed_moe_forward_fused_q8_1_input`,
   `indexed_moe_fused_decode`, `grouped_moe_gemm_prequantized`,
   `supports_moe_grouped`), `fast_mmvq.rs`.
6. **CUDA MMQ** — the fork's `mmq_gguf/mmq_gguf.cuh` already carries the tile
   sizes, traits and `vec_dot`s for most types; what is usually missing is the
   codebook: `mmq_common.cuh` stubs the iq grids to zero. Add the real table
   (share it with the MoE kernels via `kernels/iq<name>_grid.cuh`), copy
   `mmq_instance_iq4_xs.cu` for the new type, and wire `fast_mmq.rs`
   (`supports`, `qk_for`, `ds_layout_for`, `mmq_launcher`, `mmq_moe_launcher`).
   A type whose tile layout matches Q8_0's can reuse Q8_0's vec-dot (that is what
   Q2_0 does: only `load_tiles_q2_0` is type-specific).
7. **Tests** — CPU oracle tests (and AVX2-vs-scalar) plus CUDA tests against the
   CPU kernel: indexed MoE, dense MMVQ, chunked MMVQ, fused decode, grouped
   prefill, dense MMQ, grouped MMQ.

## Notes

- A `block_iq4_nl`'s `qs` sits at byte offset 2, so the CUDA code must use the
  2-byte-safe `get_int_from_uint8`, not the 4-byte `..._aligned` variant (the
  latter faults with `CUDA_ERROR_MISALIGNED_ADDRESS`). `block_iq4_xs::qs` is at
  offset 8 and is fine either way; `block_q2_0::qs` is read byte-wise.
- The grid types' 2-bit/3-bit codes are unpacked in the kernel, so keep them in
  the *tile loader* (`load_tiles_*`), not the vec-dot, when a Q8_0-compatible
  layout can be produced. `q2_0_expand` expands one Q2_0 byte (four 2-bit codes)
  into four signed bytes `-1/0/1/2`; note the codes are packed two-per-nibble, so
  a single `__byte_perm` cannot expand a byte directly (that mistake showed up as
  garbage and is worth remembering).
- IQ2_S's 1024-entry `u64` grid, IQ3_XXS's 256-entry `u32` grid and IQ3_S's
  512-entry `u32` grid live once in `kernels/iq2s_grid.cuh`,
  `kernels/iq3xxs_grid.cuh` and `kernels/iq3s_grid.cuh` and are `#include`d by
  both the MoE kernels and `mmq_common.cuh`.
- `supports_mmq(dtype)` being true makes the MoE prefill use the MMQ pair path;
  `grouped_moe_gemm_prequantized` (`moe_grouped.cu`) is the fallback and is also
  wired and tested.
- `fast_mmq::grouped` (and the MoE paths generally) take `xs` with one row per
  *assignment* (`total_assignments` rows), and `ids_src`/`ids_dst` are the
  dispatch's `sorted_token_ids`. Passing only the token rows or the wrong id
  array silently produces wrong/zero output rows -- the grouped MMQ tests spell
  this out.
- A dense batch over 8 without an MMQ kernel chunks through MMVQ
  (`GgufMatMul::mmvq_chunked`).
- To use this from `naiwa-rs`, push the fork and the candle fork and move the
  pinned `mistralrs` rev; `Cargo.toml` here points candle at `../candle`.
