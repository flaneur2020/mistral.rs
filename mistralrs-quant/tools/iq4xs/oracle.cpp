// Oracle generator for the IQ4_XS CPU-kernel unit tests.
//
// Links against the ggml objects built in ~/requant/obj, so every "expected"
// value below comes from ggml itself (dequantize_row_iq4_xs,
// quantize_row_iq4_xs_ref, quantize_row_q8_K_ref), never from a second
// re-implementation.
//
//   g++ -O2 -std=c++17 oracle.cpp ~/requant/obj/*.o -o oracle
//   python3 extract_real.py            # writes real_iq4xs.bin
//   ./oracle iq4xs_oracle.bin
//
// Output layout (all integers little-endian):
//   magic "IQ4XSOR1"            8 bytes
//   u32 nb_real, nb_syn, nq, 0
//   real_raw   nb_real * 136      packed IQ4_XS blocks from the real model
//   real_deq   nb_real * 256 f32  ggml dequantization of real_raw
//   real_act   nb_real * 256 f32  a random activation row per block
//   real_q8    nb_real * 292      quantize_row_q8_K_ref of real_act
//   real_dot   nb_real f32        f64 dot of the two dequantized sides
//   syn_raw    nb_syn  * 136      hand-built blocks (adversarial headers)
//   syn_deq    nb_syn  * 256 f32
//   syn_act    nb_syn  * 256 f32
//   syn_q8     nb_syn  * 292
//   syn_dot    nb_syn  f32
//   quant_in   nq * 256 f32       input to quantize_row_iq4_xs_ref
//   quant_out  nq * 136           its output
//   quant_deq  nq * 256 f32       dequantization of quant_out
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

typedef uint16_t ggml_half;
typedef struct {
    ggml_half d;
    uint16_t  scales_h;
    uint8_t   scales_l[4];
    uint8_t   qs[128];
} block_iq4_xs;
typedef struct {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
} block_q8_K;

extern "C" {
void dequantize_row_iq4_xs(const block_iq4_xs * x, float * y, int64_t k);
void quantize_row_iq4_xs_ref(const float * x, block_iq4_xs * y, int64_t k);
void quantize_row_q8_K_ref(const float * x, block_q8_K * y, int64_t k);
void dequantize_row_q8_K(const block_q8_K * x, float * y, int64_t k);
}

static uint32_t rng_state = 0x12345678u;
static uint32_t rnd() { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rndf() { return (float)((int32_t)(rnd() >> 8) % 2000001 - 1000000) / 100000.0f; }

template <typename T>
static void put(std::vector<uint8_t> & out, const T * p, size_t n) {
    const uint8_t * b = (const uint8_t *)p;
    out.insert(out.end(), b, b + n * sizeof(T));
}
static void put_u32(std::vector<uint8_t> & out, uint32_t v) { put(out, &v, 1); }

// For a bank of `nb` packed blocks: ggml dequantization, a random activation row
// per block, its Q8_K quantization, and the reference dot (f64 of the two
// dequantized sides).
static void bank(const uint8_t * raw, int nb, std::vector<uint8_t> & out) {
    std::vector<float> deq((size_t)nb * 256);
    dequantize_row_iq4_xs((const block_iq4_xs *)raw, deq.data(), (int64_t)nb * 256);
    for (float v : deq) if (!std::isfinite(v)) { fprintf(stderr, "non-finite dequant\n"); exit(1); }

    std::vector<float> act((size_t)nb * 256), act_deq(256);
    std::vector<int8_t> q8((size_t)nb * 292);
    std::vector<float> dots(nb);
    for (int b = 0; b < nb; ++b) {
        for (int i = 0; i < 256; ++i) act[(size_t)b * 256 + i] = rndf();
        block_q8_K q;
        quantize_row_q8_K_ref(act.data() + (size_t)b * 256, &q, 256);
        memcpy(q8.data() + (size_t)b * 292, &q, 292);
        dequantize_row_q8_K(&q, act_deq.data(), 256);
        double s = 0;
        for (int i = 0; i < 256; ++i)
            s += (double)deq[(size_t)b * 256 + i] * (double)act_deq[i];
        dots[b] = (float)s;
    }
    put(out, raw, (size_t)nb * 136);
    put(out, deq.data(), deq.size());
    put(out, act.data(), act.size());
    put(out, q8.data(), q8.size());
    put(out, dots.data(), dots.size());
}

int main(int argc, char ** argv) {
    const int NB_REAL = 16;
    const int NB_SYN  = 32;
    const int NQ      = 4;

    std::vector<uint8_t> real_raw(NB_REAL * 136);
    {
        FILE * f = fopen("real_iq4xs.bin", "rb");
        if (!f) { fprintf(stderr, "run extract_real.py first\n"); return 1; }
        if (fread(real_raw.data(), 1, real_raw.size(), f) != real_raw.size()) return 1;
        fclose(f);
    }

    std::vector<uint8_t> syn_raw(NB_SYN * 136);
    for (int b = 0; b < NB_SYN; ++b) {
        block_iq4_xs blk;
        switch (b % 6) {
            case 0: blk.d = (ggml_half)0x0000; break;             // +0.0
            case 1: blk.d = (ggml_half)0x8000; break;             // -0.0
            case 2: blk.d = (ggml_half)0x3c00; break;             // +1.0
            case 3: blk.d = (ggml_half)0xbc00; break;             // -1.0
            case 4: blk.d = (ggml_half)0x0001; break;             // min subnormal
            default: blk.d = (ggml_half)(rnd() & 0x7bff); break;  // finite positive
        }
        blk.scales_h = (uint16_t)(rnd() & 0xffff);
        for (int i = 0; i < 4; ++i) blk.scales_l[i] = (uint8_t)(rnd() & 0xff);
        for (int i = 0; i < 128; ++i) blk.qs[i] = (uint8_t)(rnd() & 0xff);
        memcpy(syn_raw.data() + b * 136, &blk, 136);
    }

    std::vector<float> qin(NQ * 256), qdeq(NQ * 256);
    for (float & v : qin) v = rndf();
    std::vector<uint8_t> qout(NQ * 136);
    quantize_row_iq4_xs_ref(qin.data(), (block_iq4_xs *)qout.data(), NQ * 256);
    dequantize_row_iq4_xs((const block_iq4_xs *)qout.data(), qdeq.data(), NQ * 256);

    std::vector<uint8_t> out;
    const char magic[8] = {'I','Q','4','X','S','O','R','1'};
    out.insert(out.end(), magic, magic + 8);
    put_u32(out, NB_REAL); put_u32(out, NB_SYN); put_u32(out, NQ); put_u32(out, 0);
    bank(real_raw.data(), NB_REAL, out);
    bank(syn_raw.data(), NB_SYN, out);
    put(out, qin.data(), qin.size());
    put(out, qout.data(), qout.size());
    put(out, qdeq.data(), qdeq.size());

    FILE * o = fopen(argc > 1 ? argv[1] : "iq4xs_oracle.bin", "wb");
    fwrite(out.data(), 1, out.size(), o);
    fclose(o);
    printf("wrote %zu bytes\n", out.size());
    return 0;
}
