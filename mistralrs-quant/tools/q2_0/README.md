# Q2_0 oracle

Unit-test data for the Q2_0 CPU kernels (`src/gguf/q2_0.rs`). Every expected
value is produced by **ggml itself**, so the tests check the Rust kernel against
the reference implementation, not against a second re-implementation.

`src/gguf/testdata/q2_0_oracle.bin` holds, for a bank of real model blocks and a
bank of hand-built blocks: ggml's dequantization, a random activation row per
block, its `quantize_row_q8_0_ref` output (two Q8_0 blocks per 64-element Q2_0
block), and the reference dot; plus an input/output pair for the Q2_0 quantizer.
The layout is documented at the top of `oracle.cpp`.

## Regenerate

```sh
# 1. 16 packed Q2_0 blocks of blk.1.ffn_down_exps.weight from the Coder GGUF
python3 tools/q2_0/extract_real.py \
    ~/models/qwen3-coder/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
    /tmp/real_q2_0.bin

# 2. link the oracle against the ggml objects built for ~/requant
g++ -O2 -std=c++17 tools/q2_0/oracle.cpp ~/requant/obj/*.o -o /tmp/q2_0_oracle

# 3. run it from a directory holding real_q2_0.bin
cd /tmp && /tmp/q2_0_oracle q2_0_oracle.bin
cp /tmp/q2_0_oracle.bin src/gguf/testdata/q2_0_oracle.bin
```

`extract_real.py` assumes the ggml default 32-byte data alignment; this GGUF has
no `general.alignment` key. ggml has no SIMD kernel for Q2_0 (its CPU dot is the
generic one), so the Rust kernel here is scalar too.
