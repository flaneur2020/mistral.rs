# IQ3_XXS oracle

Unit-test data for the IQ3_XXS CPU **read path** (`src/gguf/iq3xxs.rs`). Every
expected value is produced by **ggml itself**, so the tests check the Rust
dequantizer and vec-dot against the reference implementation.

`src/gguf/testdata/iq3xxs_oracle.bin` holds, for a bank of real model super-blocks
and a bank of hand-built ones: ggml's dequantization, a random activation row per
block, its `quantize_row_q8_K_ref` output, and the reference dot.

There is deliberately **no quantizer oracle**: the Rust quantizer is a plain
search over `IQ3XXS_GRID`, not ggml's precomputed 12-bit map plus neighbour
search, so its output is valid but not byte-identical. Only its round-trip error
is tested, and nothing in the load path calls it.

## Regenerate

```sh
# 1. 16 packed IQ3_XXS blocks of blk.0.ffn_gate_exps.weight from the Coder GGUF
python3 tools/iq3xxs/extract_real.py \
    ~/models/qwen3-coder/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
    /tmp/real_iq3xxs.bin

# 2. link the oracle against the ggml objects built for ~/requant
g++ -O2 -std=c++17 tools/iq3xxs/oracle.cpp ~/requant/obj/*.o -o /tmp/iq3xxs_oracle

# 3. run it from a directory holding real_iq3xxs.bin
cd /tmp && /tmp/iq3xxs_oracle iq3xxs_oracle.bin
cp /tmp/iq3xxs_oracle.bin src/gguf/testdata/iq3xxs_oracle.bin
```

`src/gguf/iq3xxs/tables.rs` (the grid, sign and mask tables) is generated from
`ggml-common.h`; keep it in sync with the same commit as the oracle.
