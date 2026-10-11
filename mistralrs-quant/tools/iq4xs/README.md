# IQ4_XS oracle

Unit-test data for the IQ4_XS CPU kernels (`src/gguf/iq4xs.rs`). Every
expected value is produced by **ggml itself**, so the tests check the Rust
kernel against the reference implementation, not against a second
re-implementation.

`src/gguf/testdata/iq4xs_oracle.bin` holds, for a bank of real model blocks
and a bank of hand-built blocks: ggml's dequantization, a random activation row
per block, its `quantize_row_q8_K_ref` output, and the reference dot; plus an
input/output pair for `quantize_row_iq4_xs_ref`. The layout is documented at the
top of `oracle.cpp`.

## Regenerate

```sh
# 1. 16 packed IQ4_XS blocks of blk.47.ffn_gate_exps.weight from the Coder GGUF
python3 tools/iq4xs/extract_real.py \
    ~/models/qwen3-coder/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
    /tmp/real_iq4xs.bin

# 2. link the oracle against the ggml objects built for ~/requant
g++ -O2 -std=c++17 tools/iq4xs/oracle.cpp ~/requant/obj/*.o -o /tmp/iq4xs_oracle

# 3. run it from a directory holding real_iq4xs.bin
cp /tmp/real_iq4xs.bin /tmp && (cd /tmp && /tmp/iq4xs_oracle iq4xs_oracle.bin)
cp /tmp/iq4xs_oracle.bin src/gguf/testdata/iq4xs_oracle.bin
```

`extract_real.py` assumes the ggml default 32-byte data alignment; this GGUF has
no `general.alignment` key. The generated `d` fields are around 1e-5, which is
the expected super-block scale for IQ4_XS (the effective per-group scale is
`d * (ls - 32)`, up to ~31x larger).
