# IQ2_S oracle

Unit-test data for the IQ2_S CPU **read path** (`src/gguf/iq2s.rs`). Every
expected value is produced by **ggml itself**, so the tests check the Rust
dequantizer and vec-dot against the reference implementation.

`src/gguf/testdata/iq2_s_oracle.bin` holds, for a bank of real model super-blocks
and a bank of hand-built ones: ggml's dequantization, a random activation row per
block, its `quantize_row_q8_K_ref` output, and the reference dot.

There is deliberately **no quantizer oracle**: the Rust quantizer is a plain
search over `IQ2S_GRID`, not ggml's precomputed map plus neighbour search, so its
output is valid but not byte-identical. Only its round-trip error is tested, and
nothing in the load path calls it.

## Regenerate

```sh
# 1. 16 packed IQ2_S blocks of blk.1.ffn_gate_exps.weight from the Coder GGUF
python3 tools/iq2s/extract_real.py \
    ~/models/qwen3-coder/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
    /tmp/real_iq2_s.bin

# 2. link the oracle against the ggml objects built for ~/requant
g++ -O2 -std=c++17 tools/iq2s/oracle.cpp ~/requant/obj/*.o -o /tmp/iq2s_oracle

# 3. run it from a directory holding real_iq2_s.bin
cd /tmp && /tmp/iq2s_oracle iq2_s_oracle.bin
cp /tmp/iq2_s_oracle.bin src/gguf/testdata/iq2_s_oracle.bin
```

`src/gguf/iq2s/tables.rs` (the `iq2s_grid` codebook and the sign mask) is
generated from `ggml-common.h`; `kernels/iq2s_grid.cuh` is the same table for
CUDA. Keep both in sync with the same commit as the oracle.
