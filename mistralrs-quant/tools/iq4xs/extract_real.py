#!/usr/bin/env python3
"""Extract the first 16 IQ4_XS blocks of blk.47.ffn_gate_exps.weight from the
Qwen3.8-Flash-Next Coder GGUF, for the ggml oracle."""
import struct, sys

P = sys.argv[1] if len(sys.argv) > 1 else (
    '/home/yazhou/models/qwen3-coder/IQ1_M/'
    'Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf')
OUT = sys.argv[2] if len(sys.argv) > 2 else 'real_iq4xs.bin'
N = 16

f = open(P, 'rb')
assert f.read(4) == b'GGUF'
struct.unpack('<I', f.read(4))
nt, = struct.unpack('<Q', f.read(8))
nkv, = struct.unpack('<Q', f.read(8))
FIX = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

def rs():
    n, = struct.unpack('<Q', f.read(8))
    return f.read(n)

for _ in range(nkv):
    rs()
    t, = struct.unpack('<I', f.read(4))
    if t == 8:
        rs()
    elif t == 9:
        et, = struct.unpack('<I', f.read(4))
        n, = struct.unpack('<Q', f.read(8))
        if et == 8:
            for _ in range(n):
                rs()
        else:
            f.read(n * FIX[et])
    elif t in (0, 1, 7):
        f.read(1)
    elif t in (2, 3):
        f.read(2)
    elif t in (4, 5, 6):
        f.read(4)
    else:
        f.read(8)

target = None
for _ in range(nt):
    name = rs().decode()
    nd, = struct.unpack('<I', f.read(4))
    struct.unpack('<%dQ' % nd, f.read(8 * nd))
    dt, = struct.unpack('<I', f.read(4))
    off, = struct.unpack('<Q', f.read(8))
    if name == 'blk.47.ffn_gate_exps.weight':
        assert dt == 23, dt
        target = off
assert target is not None

# data section starts at the alignment boundary after the tensor infos
end = f.tell()
align = 32  # ggml default; this file has no general.alignment key
data_off = (end + align - 1) // align * align

f.seek(data_off + target)
raw = f.read(136 * N)
assert len(raw) == 136 * N
open(OUT, 'wb').write(raw)

def f16(b):
    h, = struct.unpack('<H', b)
    return struct.unpack('<e', struct.pack('<H', h))[0]

print(f'data_off={data_off} tensor_off={target} wrote {len(raw)} bytes')
print('block d values:', [round(f16(raw[i * 136:i * 136 + 2]), 8) for i in range(4)])
