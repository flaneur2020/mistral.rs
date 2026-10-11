# IQ4_NL oracle

Unit-test data for the IQ4_NL CPU kernels (`src/gguf/iq4nl.rs`). Every expected
value is produced by **ggml itself**, so the tests check the Rust kernel against
the reference implementation, not against a second re-implementation.

`src/gguf/testdata/iq4nl_oracle.bin` holds, for a bank of real model blocks and a
bank of hand-built blocks: ggml's dequantization, a random activation row per
block, its `quantize_row_q8_0_ref` output, and the reference dot; plus an
input/output pair for the IQ4_NL quantizer. The layout is documented at the top
of `oracle.cpp`.

## Regenerate

```sh
# 1. 16 packed IQ4_NL blocks of blk.0.ffn_down_exps.weight from the Coder GGUF
python3 tools/iq4nl/extract_real.py \
    ~/models/qwen3-coder/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
    /tmp/real_iq4nl.bin

# 2. link the oracle against the ggml objects built for ~/requant
g++ -O2 -std=c++17 tools/iq4nl/oracle.cpp ~/requant/obj/*.o -o /tmp/iq4nl_oracle

# 3. run it from a directory holding real_iq4nl.bin
cd /tmp && /tmp/iq4nl_oracle iq4nl_oracle.bin
cp /tmp/iq4nl_oracle.bin src/gguf/testdata/iq4nl_oracle.bin
```

`extract_real.py` assumes the ggml default 32-byte data alignment; this GGUF has
no `general.alignment` key. The oracle quantizer is `quantize_iq4_nl` (the model's
own, `ntry = 7`), not the `ntry = -1` `quantize_row_iq4_nl_ref` wrapper.
