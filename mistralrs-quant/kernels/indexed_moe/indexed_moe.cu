// Indexed MoE forward kernel for quantized weights
// Adapted from llama.cpp ggml-cuda.cu and candle-kernels
// https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda.cu

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#include "../iq2s_grid.cuh"
#include "../iq3s_grid.cuh"

#define GGML_UNUSED(x) (void)(x)

#define QK_K 256
#define K_SCALE_SIZE 12

#define WARP_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define K_QUANTS_PER_ITERATION 2

typedef uint16_t ggml_fp16_t;

// Helper functions
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, 32);
  }
  return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
  }
  return x;
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t *x8,
                                                        const int &i32) {
  const uint16_t *x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;
  return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t *x8,
                                                         const int &i32) {
  const uint16_t *x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;
  return x32;
}

static __device__ __forceinline__ int
get_int_from_int8_aligned(const int8_t *x8, const int &i32) {
  return *((const int *)(x8 + sizeof(int) * i32));
}

static __device__ __forceinline__ int
get_int_from_uint8_aligned(const uint8_t *x8, const int &i32) {
  return *((const int *)(x8 + sizeof(int) * i32));
}

#define MIN_CC_DP4A 610

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b,
                                                     int c) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A
  return __dp4a(a, b, c);
#else
  const int8_t *a8 = (const int8_t *)&a;
  const int8_t *b8 = (const int8_t *)&b;
  return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

// Block type definitions

// The 16-entry non-linear IQ4_XS/IQ4_NL codebook.
static __device__ __align__(16) const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// `q4` holds 8 four-bit indices; select those bytes from `table` (which must be
// 16-byte aligned). The first int gets the even indices, the second the odd.
static __device__ __forceinline__ int2
 get_int_from_table_16(const int &q4, const int8_t *table) {
  const uint32_t *table32 = (const uint32_t *)table;
  uint32_t tmp[2];
  const uint32_t low_high_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
    const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
    tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
  }
  return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420),
                   __byte_perm(tmp[0], tmp[1], 0x7531));
}

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
  half d;
  int8_t qs[QK8_0];
} block_q8_0;

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct {
  half2 ds;
  int8_t qs[QK8_0];
} block_q8_1;

#define QR2_K 4
#define QI2_K (QK_K / (4 * QR2_K))
typedef struct {
  uint8_t scales[QK_K / 16];
  uint8_t qs[QK_K / 4];
  half2 dm;
} block_q2_K;

#define QR3_K 4
#define QI3_K (QK_K / (4 * QR3_K))
typedef struct {
  uint8_t hmask[QK_K / 8];
  uint8_t qs[QK_K / 4];
  uint8_t scales[K_SCALE_SIZE];
  half d;
} block_q3_K;

#define QR4_K 2
#define QI4_K (QK_K / (4 * QR4_K))
typedef struct {
  half2 dm;
  uint8_t scales[3 * QK_K / 64];
  uint8_t qs[QK_K / 2];
} block_q4_K;

#define QR5_K 2
#define QI5_K (QK_K / (4 * QR5_K))
typedef struct {
  half2 dm;
  uint8_t scales[K_SCALE_SIZE];
  uint8_t qh[QK_K / 8];
  uint8_t qs[QK_K / 2];
} block_q5_K;

#define QR6_K 2
#define QI6_K (QK_K / (4 * QR6_K))
typedef struct {
  uint8_t ql[QK_K / 2];
  uint8_t qh[QK_K / 4];
  int8_t scales[QK_K / 16];
  half d;
} block_q6_K;

// IQ4_XS block type (GGML type 23): eight 32-element groups, each with a
// 6-bit signed scale and 4-bit codebook indices.
typedef struct {
  half d;
  uint16_t scales_h;
  uint8_t scales_l[QK_K / 64];
  uint8_t qs[QK_K / 2];
} block_iq4_xs;

// IQ4_NL (GGML type 20): one f16 scale and 16 bytes of nibbles.
#define QK4_NL 32
typedef struct {
  half d;
  uint8_t qs[QK4_NL / 2];
} block_iq4_nl;

// Q2_0 (GGML type 42): one f16 scale, 16 bytes of two-bit codes.
#define QK2_0 64
typedef struct {
  half d;
  uint8_t qs[QK2_0 / 4];
} block_q2_0;

// IQ3_XXS (GGML type 18): 256-element super-block; 64 grid-index bytes then 32
// scale/sign bytes.
typedef struct {
  half d;
  uint8_t qs[3 * QK_K / 8];
} block_iq3_xxs;

// IQ2_S (GGML type 22).
typedef struct {
  half d;
  uint8_t qs[QK_K / 4];
  uint8_t qh[QK_K / 32];
  uint8_t scales[QK_K / 32];
} block_iq2_s;

// IQ3_S (GGML type 21).
typedef struct {
  half d;
  uint8_t qs[QK_K / 4];
  uint8_t qh[QK_K / 32];
  uint8_t signs[QK_K / 8];
  uint8_t scales[QK_K / 64];
} block_iq3_s;




// Q4_0 block type
#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
  half d;
  uint8_t qs[QK4_0 / 2];
} block_q4_0;

// Q4_1 block type
#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
  half2 dm;
  uint8_t qs[QK4_1 / 2];
} block_q4_1;

// Q5_0 block type
#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
  half d;
  uint8_t qh[4];
  uint8_t qs[QK5_0 / 2];
} block_q5_0;

// Q5_1 block type
#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
  half2 dm;
  uint8_t qh[4];
  uint8_t qs[QK5_1 / 2];
} block_q5_1;

// VDR constants
#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_1_Q8_1_MMVQ 2
#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_IQ4_XS_Q8_1_MMVQ 1
#define VDR_IQ4_NL_Q8_1_MMVQ 1
#define VDR_Q2_0_Q8_1_MMVQ 1
#define QI3_XXS 16
#define VDR_IQ3_XXS_Q8_1_MMVQ 2
#define QI2_S 16
#define VDR_IQ2_S_Q8_1_MMVQ 2
#define QI3_S 16
#define VDR_IQ3_S_Q8_1_MMVQ 2

// vec_dot implementations for Q4_0, Q4_1, Q5_0, Q5_1

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q4_0_q8_1_impl(const int *v, const int *u, const float &d4,
                       const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
    const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 ds8f = __half22float2(ds8);
  return d4 * (sumi * ds8f.x - (8 * vdr / QI4_0) * ds8f.y);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q4_1_q8_1_impl(const int *v, const int *u, const half2 &dm4,
                       const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
    const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 dm4f = __half22float2(dm4);
  const float2 ds8f = __half22float2(ds8);
  const float d4d8 = dm4f.x * ds8f.x;
  const float m4s8 = dm4f.y * ds8f.y;
  return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q5_0_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const float &d5, const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
    vi0 |= (vh[i] << 4) & 0x00000010;
    vi0 |= (vh[i] << 11) & 0x00001000;
    vi0 |= (vh[i] << 18) & 0x00100000;
    vi0 |= (vh[i] << 25) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);

    int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
    vi1 |= (vh[i] >> 12) & 0x00000010;
    vi1 |= (vh[i] >> 5) & 0x00001000;
    vi1 |= (vh[i] << 2) & 0x00100000;
    vi1 |= (vh[i] << 9) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 ds8f = __half22float2(ds8);
  return d5 * (sumi * ds8f.x - (16 * vdr / QI5_0) * ds8f.y);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q5_1_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const half2 &dm5, const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
    vi0 |= (vh[i] << 4) & 0x00000010;
    vi0 |= (vh[i] << 11) & 0x00001000;
    vi0 |= (vh[i] << 18) & 0x00100000;
    vi0 |= (vh[i] << 25) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);

    int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
    vi1 |= (vh[i] >> 12) & 0x00000010;
    vi1 |= (vh[i] >> 5) & 0x00001000;
    vi1 |= (vh[i] << 2) & 0x00100000;
    vi1 |= (vh[i] << 9) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 dm5f = __half22float2(dm5);
  const float2 ds8f = __half22float2(ds8);
  const float d5d8 = dm5f.x * ds8f.x;
  const float m5s8 = dm5f.y * ds8f.y;
  return sumi * d5d8 + m5s8 / (QI5_1 / vdr);
}

// vec_dot implementations for Q8_0 and K-quants

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q8_0_q8_1_impl(const int *v, const int *u, const half &d8_0,
                       const half &d8_1) {

  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
  }
  return sumi * __half2float(d8_0) * __half2float(d8_1);
}

static __device__ __forceinline__ float
vec_dot_q2_K_q8_1_impl_mmvq(const int &v, const int *__restrict__ u,
                            const uint8_t *__restrict__ scales,
                            const half2 &dm2, const float *__restrict__ d8) {

  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

#pragma unroll
  for (int i = 0; i < QR2_K; ++i) {
    const int sc = scales[2 * i];
    const int vi = (v >> (2 * i)) & 0x03030303;
    sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));
    int m = sc >> 4;
    m |= m << 8;
    m |= m << 16;
    sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);
  }
  const float2 dm2f = __half22float2(dm2);
  return dm2f.x * sumf_d - dm2f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int &vl, const int &vh, const int *__restrict__ u,
    const uint8_t *__restrict__ scales, const int &scale_offset,
    const float &d3, const float *__restrict__ d8) {

  float sumf = 0.0f;

#pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    const int isc = scale_offset + 2 * i;
    const int isc_low = isc % (QK_K / 32);
    const int sc_shift_low = 4 * (isc / (QK_K / 32));
    const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;
    const int isc_high = isc % (QK_K / 64);
    const int sc_shift_high = 2 * (isc / (QK_K / 64));
    const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3)
                        << 4;
    const int sc = (sc_low | sc_high) - 32;
    const int vil = (vl >> (2 * i)) & 0x03030303;
    const int vih = ((vh >> i) << 2) & 0x04040404;
    const int vi = __vsubss4(vil, vih);
    sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
  }
  return d3 * sumf;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const half2 &dm4, const float *__restrict__ d8) {

  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 =
        ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
    const int dot2 = ggml_cuda_dp4a(
        0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm4f = __half22float2(dm4);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const half2 &dm5,
    const float *__restrict__ d8) {

  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
    const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
    const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
    const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
    const int v0i = vl0i | vh0i;
    const int v1i = vl1i | vh1i;
    const int dot1 =
        ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
    const int dot2 = ggml_cuda_dp4a(
        0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm5f = __half22float2(dm5);
  return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

static __device__ __forceinline__ float
vec_dot_q6_K_q8_1_impl_mmvq(const int &vl, const int &vh,
                            const int *__restrict__ u,
                            const int8_t *__restrict__ scales, const float &d,
                            const float *__restrict__ d8) {

  float sumf = 0.0f;

#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    const int sc = scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = __vsubss4((vil | vih), 0x20202020);
    sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
  }
  return d * sumf;
}

// vec_dot wrapper functions
typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ vbq,
                                  const block_q8_1 *__restrict__ bq8_1,
                                  const int &iqs);

static __device__ __forceinline__ float
vec_dot_q4_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q4_0 *bq4_0 = (const block_q4_0 *)vbq;
  int v[VDR_Q4_0_Q8_1_MMVQ];
  int u[2 * VDR_Q4_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_uint8(bq4_0->qs, iqs + i);
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
  }
  return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q4_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q4_1 *bq4_1 = (const block_q4_1 *)vbq;
  int v[VDR_Q4_1_Q8_1_MMVQ];
  int u[2 * VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
  }
  return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q5_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q5_0 *bq5_0 = (const block_q5_0 *)vbq;
  int vl[VDR_Q5_0_Q8_1_MMVQ];
  int vh[VDR_Q5_0_Q8_1_MMVQ];
  int u[2 * VDR_Q5_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
    vl[i] = get_int_from_uint8(bq5_0->qs, iqs + i);
    vh[i] = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
  }
  return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d,
                                                    bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q5_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q5_1 *bq5_1 = (const block_q5_1 *)vbq;
  int vl[VDR_Q5_1_Q8_1_MMVQ];
  int vh[VDR_Q5_1_Q8_1_MMVQ];
  int u[2 * VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
    vl[i] = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
    vh[i] = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
  }
  return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm,
                                                    bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q8_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1_v, const int &iqs) {
  const block_q8_1 *bq8_1_w = (const block_q8_1 *)vbq;
  int v[VDR_Q8_1_Q8_1_MMVQ];
  int u[VDR_Q8_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q8_1_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8_aligned(bq8_1_w->qs, iqs + i);
    u[i] = get_int_from_int8_aligned(bq8_1_v->qs, iqs + i);
  }
  const float2 dmw = __half22float2(bq8_1_w->ds);
  const float2 dmv = __half22float2(bq8_1_v->ds);
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < VDR_Q8_1_Q8_1_MMVQ; ++i) {
    sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
  }
  return dmw.x * dmv.x * sumi + dmw.y * dmv.y;
}

static __device__ __forceinline__ float
vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q8_0 *bq8_0 = (const block_q8_0 *)vbq;
  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
    u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
  }
  return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                    __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float
vec_dot_q2_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q2_K *bq2_K = (const block_q2_K *)vbq;
  const int bq8_offset = QR2_K * (iqs / QI8_1);
  const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
  const uint8_t *scales = bq2_K->scales + scale_offset;
  const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
  int u[QR2_K];
  float d8[QR2_K];

#pragma unroll
  for (int i = 0; i < QR2_K; ++i) {
    u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
    d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
  }
  return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float
vec_dot_q3_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q3_K *bq3_K = (const block_q3_K *)vbq;
  const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
  const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
  const float d = bq3_K->d;
  const int vl = get_int_from_uint8(bq3_K->qs, iqs);
  const int vh =
      ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;
  int u[QR3_K];
  float d8[QR3_K];

#pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
    d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
  }
  return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d,
                                     d8);
}

static __device__ __forceinline__ float
vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q4_K *bq4_K = (const block_q4_K *)vbq;
  int v[2];
  int u[2 * QR4_K];
  float d8[QR4_K];

  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int *q4 =
      (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];

  const uint16_t *scales = (const uint16_t *)bq4_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;

  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

// IQ4_XS: one 32-element group per call. `iqs` is the group index (0..7), so
// the caller only needs 8 cooperating threads per weight super-block.
// Ported from `ggml-cuda/vecdotq.cuh::vec_dot_iq4_xs_q8_1`.
static __device__ __forceinline__ float
vec_dot_iq4_xs_q8_1(const void *__restrict__ vbq,
                    const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_iq4_xs *bq4 = (const block_iq4_xs *)vbq;

  int sumi = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int aux_q4 = get_int_from_uint8_aligned(bq4->qs, 4 * iqs + j);
    const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

    const int u0 = get_int_from_int8_aligned(bq8_1[iqs].qs, j + 0);
    const int u1 = get_int_from_int8_aligned(bq8_1[iqs].qs, j + 4);

    sumi = ggml_cuda_dp4a(v.x, u0, sumi);
    sumi = ggml_cuda_dp4a(v.y, u1, sumi);
  }

  const int ls =
      ((bq4->scales_l[iqs / 2] >> (4 * (iqs % 2))) & 0x0f) |
      (((bq4->scales_h >> (2 * iqs)) & 0x03) << 4);
  sumi *= ls - 32;

  const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs].ds);
  return d * sumi;
}

// IQ4_NL: the whole 32-element block per call (qi = vdr = 1), so `iqs` is 0.
// Ported from `ggml-cuda/vecdotq.cuh::vec_dot_iq4_nl_q8_1`.
static __device__ __forceinline__ float
vec_dot_iq4_nl_q8_1(const void *__restrict__ vbq,
                    const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  (void)iqs;
  const block_iq4_nl *bq4 = (const block_iq4_nl *)vbq;
  int sumi = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int aux_q4 = get_int_from_uint8(bq4->qs, j);
    const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
    const int u0 = get_int_from_int8_aligned(bq8_1->qs, j + 0);
    const int u1 = get_int_from_int8_aligned(bq8_1->qs, j + 4);
    sumi = ggml_cuda_dp4a(v.x, u0, sumi);
    sumi = ggml_cuda_dp4a(v.y, u1, sumi);
  }
  return __half2float(bq4->d) * __low2float(bq8_1->ds) * sumi;
}

// Q2_0: one 32-element chunk per call, `iqs` selects the chunk (0/1).
static __device__ __forceinline__ float
vec_dot_q2_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q2_0 *bq2 = (const block_q2_0 *)vbq;
  const uint8_t *qs = bq2->qs + iqs * 8;
  const block_q8_1 *chunk = bq8_1 + iqs;
  const int8_t *qy = chunk->qs;
  int sumi = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int byte = qs[j];
    sumi += (((byte >> 0) & 3) - 1) * (int)qy[j * 4 + 0];
    sumi += (((byte >> 2) & 3) - 1) * (int)qy[j * 4 + 1];
    sumi += (((byte >> 4) & 3) - 1) * (int)qy[j * 4 + 2];
    sumi += (((byte >> 6) & 3) - 1) * (int)qy[j * 4 + 3];
  }
  return __half2float(bq2->d) * __low2float(chunk->ds) * sumi;
}

static __device__ __forceinline__ int iq3_get_int_b2(const void *x,
                                                     const int &i32) {
  const uint16_t *x16 = (const uint16_t *)x;
  return (x16[2 * i32 + 0] << 0) | (x16[2 * i32 + 1] << 16);
}

static __device__ __forceinline__ uint32_t iq3_unpack_ksigns(const uint8_t v) {
  const uint32_t p = __popc(v) & 1;
  return (v ^ (p << 7)) * 0x01010101u;
}

static const __device__ uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// IQ3_XXS: one 32-element group per call (`iqs` is even, 0..14).
static __device__ __forceinline__ float
vec_dot_iq3_xxs_q8_1(const void *__restrict__ vbq,
                     const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_iq3_xxs *bq3 =
      (const block_iq3_xxs *)vbq;
  const int2 q3_packed = make_int2(iq3_get_int_b2(bq3->qs, iqs),
                                    iq3_get_int_b2(bq3->qs, iqs + 1));
  const uint8_t *q3 = (const uint8_t *)&q3_packed;
  const uint32_t aux32 = iq3_get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
  int sumi = 0;
#pragma unroll
  for (int l0 = 0; l0 < 8; l0 += 2) {
    const int2 grid_pos =
        make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
    const uint32_t signs = iq3_unpack_ksigns(aux32 >> (7 * l0 / 2));
    const int signs0 = __vcmpne4(signs & 0x08040201, 0);
    const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
    const int u0 = get_int_from_int8_aligned(bq8_1[iqs / 2].qs, l0 + 0);
    const int signs1 = __vcmpne4(signs & 0x80402010, 0);
    const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
    const int u1 = get_int_from_int8_aligned(bq8_1[iqs / 2].qs, l0 + 1);
    sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
    sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
  }
  const int ls = aux32 >> 28;
  sumi = (ls * sumi + sumi / 2) / 2;
  const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
  return d * sumi;
}

// IQ2_S: one 32-element group per call (`iqs` is even, 0..14).
static __device__ __forceinline__ float
vec_dot_iq2_s_q8_1(const void *__restrict__ vbq,
                   const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_iq2_s *bq2 =
      (const block_iq2_s *)vbq;
  const int qs_packed = iq3_get_int_b2(bq2->qs, iqs / 2);
  const uint8_t *qs = (const uint8_t *)&qs_packed;
  const int qh = bq2->qh[iqs / 2];
  const int signs_packed_32 = iq3_get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
  const uint8_t *signs_packed_8 = (const uint8_t *)&signs_packed_32;
  const int ls0 = bq2->scales[iqs / 2] & 0x0F;
  const int ls1 = bq2->scales[iqs / 2] >> 4;
  int sumi0 = 0;
  int sumi1 = 0;
#pragma unroll
  for (int l0 = 0; l0 < 8; l0 += 2) {
    const int *grid_pos =
        (const int *)(iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
    const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                                     ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                                 0x00000000);
    const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                                     ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                                 0x00000000);
    const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
    const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
    const int u0 = get_int_from_int8_aligned(bq8_1[iqs / 2].qs, l0 + 0);
    const int u1 = get_int_from_int8_aligned(bq8_1[iqs / 2].qs, l0 + 1);
    if (l0 < 4) {
      sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
      sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
    } else {
      sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
      sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
    }
  }
  const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
  const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
  return d * sumi;
}

// IQ3_S: one 32-element group per call (`iqs` is even, 0..14).
static __device__ __forceinline__ float
vec_dot_iq3_s_q8_1(const void *__restrict__ vbq,
                   const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_iq3_s *bq3 =
      (const block_iq3_s *)vbq;
  const int2 qs_packed = make_int2(iq3_get_int_b2(bq3->qs, iqs + 0),
                                   iq3_get_int_b2(bq3->qs, iqs + 1));
  const uint8_t *qs = (const uint8_t *)&qs_packed;
  const int qh = bq3->qh[iqs / 2];
  const int signs_packed_32 = iq3_get_int_b2(bq3->signs, iqs / 2);
  const uint8_t *signs_packed_8 = (const uint8_t *)&signs_packed_32;
  int sumi = 0;
#pragma unroll
  for (int l0 = 0; l0 < 8; l0 += 2) {
    const int2 grid_pos = make_int2(
        iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
        iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
    const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                                     ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                                 0x00000000);
    const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                                     ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                                 0x00000000);
    const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
    const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
    const int u0 = get_int_from_int8_aligned(bq8_1[iqs / 2].qs, l0 + 0);
    const int u1 = get_int_from_int8_aligned(bq8_1[iqs / 2].qs, l0 + 1);
    sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
    sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
  }
  sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
  const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
  return d * sumi;
}

static __device__ __forceinline__ float
vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q5_K *bq5_K = (const block_q5_K *)vbq;
  int vl[2];
  int vh[2];
  int u[2 * QR5_K];
  float d8[QR5_K];

  const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
  const int *ql =
      (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  const int *qh = (const int *)(bq5_K->qh + 4 * ((iqs / 2) % 4));

  vl[0] = ql[0];
  vl[1] = ql[4];
  vh[0] = qh[0] >> bq8_offset;
  vh[1] = qh[4] >> bq8_offset;

  const uint16_t *scales = (const uint16_t *)bq5_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;

#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float
vec_dot_q6_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q6_K *bq6_K = (const block_q6_K *)vbq;
  const int bq8_offset =
      2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
  const int scale_offset =
      (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
  const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
  const int vl = get_int_from_uint8(bq6_K->ql, iqs);
  const int vh =
      get_int_from_uint8(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) +
                                        iqs % (QI6_K / 4)) >>
      vh_shift;
  const int8_t *scales = bq6_K->scales + scale_offset;
  int u[QR6_K];
  float d8[QR6_K];

#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
    d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
  }
  return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

// quantize_q8_1 kernel (F32 input)
extern "C" __global__ void quantize_q8_1(const float *__restrict__ x,
                                         void *__restrict__ vy, const int kx,
                                         const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix >= kx_padded) {
    return;
  }

  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;
  block_q8_1 *y = (block_q8_1 *)vy;

  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max(amax);
  sum = warp_reduce_sum(sum);

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  reinterpret_cast<half &>(y[ib].ds.x) = d;
  reinterpret_cast<half &>(y[ib].ds.y) = sum;
}

// quantize_q8_1 kernel (BF16 input — fuses bf16→f32 cast + quantization)
extern "C" __global__ void
quantize_q8_1_bf16(const __nv_bfloat16 *__restrict__ x, void *__restrict__ vy,
                   const int kx, const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix >= kx_padded) {
    return;
  }

  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;
  block_q8_1 *y = (block_q8_1 *)vy;

  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = ix < kx ? __bfloat162float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max(amax);
  sum = warp_reduce_sum(sum);

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  reinterpret_cast<half &>(y[ib].ds.x) = d;
  reinterpret_cast<half &>(y[ib].ds.y) = sum;
}

// quantize_q8_1 kernel (F16 input — fuses f16→f32 cast + quantization)
extern "C" __global__ void quantize_q8_1_f16(const half *__restrict__ x,
                                             void *__restrict__ vy,
                                             const int kx,
                                             const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix >= kx_padded) {
    return;
  }

  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;
  block_q8_1 *y = (block_q8_1 *)vy;

  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = ix < kx ? __half2float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max(amax);
  sum = warp_reduce_sum(sum);

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  reinterpret_cast<half &>(y[ib].ds.x) = d;
  reinterpret_cast<half &>(y[ib].ds.y) = sum;
}

// Launch wrapper for BF16 quantize
extern "C" void launch_quantize_q8_1_bf16(const void *x, void *vy, int kx,
                                          int kx_padded, int num_rows,
                                          void *stream) {
  int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  quantize_q8_1_bf16<<<grid, block, 0, s>>>((const __nv_bfloat16 *)x, vy, kx,
                                            kx_padded);
}

// Launch wrapper for F16 quantize
extern "C" void launch_quantize_q8_1_f16(const void *x, void *vy, int kx,
                                         int kx_padded, int num_rows,
                                         void *stream) {
  int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  quantize_q8_1_f16<<<grid, block, 0, s>>>((const half *)x, vy, kx, kx_padded);
}

// indexed_moe_forward template
template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void indexed_moe_forward(const void *__restrict__ all_weights,
                                    const void *__restrict__ all_inputs,
                                    const unsigned int *__restrict__ indices,
                                    float *__restrict__ all_outputs,
                                    const int n, const int k, const int batch,
                                    const int topk, const int k_padded,
                                    const int input_dim1) {

  const int current_batch = blockIdx.y;
  const int current_topk = blockIdx.z;
  const int task_id = current_batch * gridDim.z + current_topk;

  if (task_id >= gridDim.y * gridDim.z) {
    return;
  }

  const int input_idx = (input_dim1 == 1) ? current_batch : task_id;
  const unsigned int expert_id = indices[task_id];

  const size_t weight_block_size = sizeof(block_q_t);
  const size_t input_block_size = sizeof(block_q8_1);
  const size_t weight_blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t weight_expert_stride_bytes =
      (size_t)n * weight_blocks_per_row * weight_block_size;
  const size_t input_task_stride_bytes =
      (size_t)k_padded / QK8_1 * input_block_size;
  const size_t output_task_stride_elems = n;

  const void *current_input_ptr =
      (const char *)all_inputs + input_idx * input_task_stride_bytes;
  const void *current_weight_ptr =
      (const char *)all_weights + expert_id * weight_expert_stride_bytes;
  float *current_output_ptr = all_outputs + task_id * output_task_stride_elems;

  constexpr int ncols_y = 1;
  constexpr int nwarps = 4;
  constexpr int rows_per_cuda_block = 1;

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;

  if (row0 >= n) {
    return;
  }

  const int blocks_per_row_x = (k + qk - 1) / qk;
  const int blocks_per_col_y = k_padded / QK8_1;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  float tmp = 0.0f;

  const block_q_t *w = (const block_q_t *)current_weight_ptr;
  const block_q8_1 *x = (const block_q8_1 *)current_input_ptr;

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));
    tmp += vec_dot_q_cuda(&w[kbx + row0 * blocks_per_row_x], &x[kby], kqs);
  }

  __shared__ float tmp_shared[nwarps - 1][WARP_SIZE];
  if (threadIdx.y > 0) {
    tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
  }
  __syncthreads();

  if (threadIdx.y == 0) {
    for (int l = 0; l < nwarps - 1; ++l) {
      tmp += tmp_shared[l][threadIdx.x];
    }
    tmp = warp_reduce_sum(tmp);
    if (threadIdx.x == 0) {
      current_output_ptr[row0] = tmp;
    }
  }
}

// Kernel instantiations - Q quants
extern "C" __global__ void indexed_moe_forward_q4_0_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                      vec_dot_q4_0_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q4_1_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                      vec_dot_q4_1_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q5_0_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                      vec_dot_q5_0_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q5_1_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                      vec_dot_q5_1_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q8_1_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                      vec_dot_q8_1_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

// Kernel instantiations - K quants
extern "C" __global__ void indexed_moe_forward_q2k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                      vec_dot_q2_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q3k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                      vec_dot_q3_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q4k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                      vec_dot_q4_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q5k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                      vec_dot_q5_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q6k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                      vec_dot_q6_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

// IQ4_XS: qi = 8 (one thread per 32-element group), vdr = 1.
extern "C" __global__ void indexed_moe_forward_iq4xs_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, 8, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ,
                      vec_dot_iq4_xs_q8_1>(all_weights, all_inputs, indices,
                                           all_outputs, n, k, batch, topk,
                                           k_padded, input_dim1);
}

// IQ4_NL: one thread per 32-element block.
extern "C" __global__ void indexed_moe_forward_iq4nl_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK4_NL, 1, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ,
                      vec_dot_iq4_nl_q8_1>(all_weights, all_inputs, indices,
                                           all_outputs, n, k, batch, topk,
                                           k_padded, input_dim1);
}

// Q2_0: qi = 2 (two 32-element chunks per 64-element block), vdr = 1.
extern "C" __global__ void indexed_moe_forward_q2_0_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK2_0, 2, block_q2_0, VDR_Q2_0_Q8_1_MMVQ,
                      vec_dot_q2_0_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_iq3xxs_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI3_XXS, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ,
                      vec_dot_iq3_xxs_q8_1>(all_weights, all_inputs, indices,
                                            all_outputs, n, k, batch, topk,
                                            k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_iq2s_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI2_S, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ,
                      vec_dot_iq2_s_q8_1>(all_weights, all_inputs, indices,
                                          all_outputs, n, k, batch, topk,
                                          k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_iq3s_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI3_S, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ,
                      vec_dot_iq3_s_q8_1>(all_weights, all_inputs, indices,
                                          all_outputs, n, k, batch, topk,
                                          k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q8_0_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                      vec_dot_q8_0_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

// ============== C wrapper functions for FFI ==============

extern "C" void launch_quantize_q8_1(const float *x, void *vy, int kx,
                                     int kx_padded, int num_blocks_x,
                                     int num_rows, void *stream) {
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  quantize_q8_1<<<grid, block, 0, cuda_stream>>>(x, vy, kx, kx_padded);
}

extern "C" void launch_indexed_moe_forward_q2k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q2k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q3k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q3k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q4k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q4k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q5k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q5k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q6k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q6k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_iq4xs_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_iq4xs_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_iq4nl_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_iq4nl_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q2_0_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q2_0_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_iq3xxs_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_iq3xxs_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_iq2s_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_iq2s_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_iq3s_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_iq3s_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q8_0_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q8_0_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q4_0_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q4_0_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q4_1_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q4_1_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q5_0_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q5_0_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q5_1_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q5_1_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q8_1_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q8_1_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

// ============== Fused MoE decode kernels ==============
//
// These kernels are optimized for decode (batch=1, seq_len=1) by:
// 1. Fusing gate+up projections with activation+multiply into one kernel
// 2. Fusing down projection with topk_weights and expert aggregation
// (atomicAdd)
// 3. Using 2 output rows per block to halve block count
// 4. Using 1 warp (32 threads) per block for warp-only reduction (no shared
// mem)

// Activation functions
static __device__ __forceinline__ float gelu_pytorch_tanh(float x) {
  // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  float x3 = x * x * x;
  float inner = 0.7978845608028654f * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

static __device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

enum MoeOutputType : int {
  MOE_OUTPUT_F32 = 0,
  MOE_OUTPUT_F16 = 1,
  MOE_OUTPUT_BF16 = 2,
};

static __device__ __forceinline__ void
moe_store_output(void *outputs, size_t index, int output_type, float value) {
  if (output_type == MOE_OUTPUT_F32) {
    static_cast<float *>(outputs)[index] = value;
  } else if (output_type == MOE_OUTPUT_F16) {
    static_cast<half *>(outputs)[index] = __float2half_rn(value);
  } else {
    static_cast<__nv_bfloat16 *>(outputs)[index] = __float2bfloat16_rn(value);
  }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void moe_gemv_gate_up_pair_impl(
    const void *__restrict__ gate_weights, const void *__restrict__ up_weights,
    const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, void *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int num_experts, const int k_padded, const int output_type) {
  constexpr int ROWS_PER_WARP = 2;
  constexpr int NWARPS = 4;
  constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;

  const int warp_id = threadIdx.y;
  const int row0 = ROWS_PER_BLOCK * blockIdx.x + warp_id * ROWS_PER_WARP;
  const int current_topk = blockIdx.y;
  const int current_batch = blockIdx.z;

  if (row0 >= n)
    return;

  const int task_id = current_batch * topk + current_topk;
  const unsigned int expert_id = indices[task_id];
  const size_t output_base = (size_t)task_id * 2 * n;
  if (expert_id >= (unsigned int)num_experts) {
    for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
      if (threadIdx.x == 0) {
        moe_store_output(all_outputs, output_base + row0 + r, output_type,
                         0.0f);
        moe_store_output(all_outputs, output_base + n + row0 + r, output_type,
                         0.0f);
      }
    }
    return;
  }
  const size_t blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t expert_stride_bytes =
      (size_t)n * blocks_per_row * sizeof(block_q_t);
  const size_t input_stride_bytes =
      (size_t)k_padded / QK8_1 * sizeof(block_q8_1);

  const block_q8_1 *x =
      (const block_q8_1 *)((const char *)all_inputs +
                           (size_t)current_batch * input_stride_bytes);
  const block_q_t *gate_w =
      (const block_q_t *)((const char *)gate_weights +
                          (size_t)expert_id * expert_stride_bytes);
  const block_q_t *up_w =
      (const block_q_t *)((const char *)up_weights +
                          (size_t)expert_id * expert_stride_bytes);

  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
  const int blocks_per_row_x = (int)blocks_per_row;
  for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
    const int row = row0 + r;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (threadIdx.x % (qi / vdr));
      gate_sum += vec_dot_q_cuda(&gate_w[kbx + (size_t)row * blocks_per_row],
                                 &x[kby], kqs);
      up_sum += vec_dot_q_cuda(&up_w[kbx + (size_t)row * blocks_per_row],
                               &x[kby], kqs);
    }

    gate_sum = warp_reduce_sum(gate_sum);
    up_sum = warp_reduce_sum(up_sum);
    if (threadIdx.x == 0) {
      moe_store_output(all_outputs, output_base + row, output_type, gate_sum);
      moe_store_output(all_outputs, output_base + n + row, output_type, up_sum);
    }
  }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void moe_gemv_lora_down_impl(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, void *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int num_experts, const int k_padded, const int output_type) {
  constexpr int ROWS_PER_WARP = 4;
  constexpr int NWARPS = 4;
  constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;

  const int warp_id = threadIdx.y;
  const int row0 = ROWS_PER_BLOCK * blockIdx.x + warp_id * ROWS_PER_WARP;
  const int current_topk = blockIdx.y;
  const int current_batch = blockIdx.z;

  if (row0 >= n)
    return;

  const int task_id = current_batch * topk + current_topk;
  const unsigned int expert_id = indices[task_id];
  const size_t output_base = (size_t)task_id * n;
  if (expert_id >= (unsigned int)num_experts) {
    for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
      if (threadIdx.x == 0) {
        moe_store_output(all_outputs, output_base + row0 + r, output_type,
                         0.0f);
      }
    }
    return;
  }
  const size_t blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t expert_stride_bytes =
      (size_t)n * blocks_per_row * sizeof(block_q_t);
  const size_t input_stride_bytes =
      (size_t)k_padded / QK8_1 * sizeof(block_q8_1);

  const block_q8_1 *x =
      (const block_q8_1 *)((const char *)all_inputs +
                           (size_t)task_id * input_stride_bytes);
  const block_q_t *w =
      (const block_q_t *)((const char *)all_weights +
                          (size_t)expert_id * expert_stride_bytes);

  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
  const int blocks_per_row_x = (int)blocks_per_row;
  for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
    const int row = row0 + r;
    float sum = 0.0f;

    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (threadIdx.x % (qi / vdr));
      sum +=
          vec_dot_q_cuda(&w[kbx + (size_t)row * blocks_per_row], &x[kby], kqs);
    }

    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
      moe_store_output(all_outputs, output_base + row, output_type, sum);
    }
  }
}

// Fused gate+up+activation+multiply kernel template
// Computes: output = up(x) * activation(gate(x)) for each expert assignment
// Grid: (ceildiv(n, NWARPS * ROWS_PER_WARP), topk, batch)
// Block: (WARP_SIZE, NWARPS, 1)
// Each warp independently handles ROWS_PER_WARP output rows
template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void moe_gemv_fused_gate_up_impl(
    const void *__restrict__ gate_weights, const void *__restrict__ up_weights,
    const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded,
    const int act_type) { // 0=gelu_pytorch_tanh, 1=silu

  constexpr int ROWS_PER_WARP = 2;
  constexpr int NWARPS = 4;
  constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;

  const int warp_id = threadIdx.y;
  const int row0 = ROWS_PER_BLOCK * blockIdx.x + warp_id * ROWS_PER_WARP;
  const int current_topk = blockIdx.y;
  const int current_batch = blockIdx.z;

  if (row0 >= n)
    return;

  const int task_id = current_batch * topk + current_topk;
  const unsigned int expert_id = indices[task_id];

  const size_t weight_block_size = sizeof(block_q_t);
  const size_t input_block_size = sizeof(block_q8_1);
  const size_t blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t expert_stride_bytes =
      (size_t)n * blocks_per_row * weight_block_size;
  const size_t input_stride_bytes = (size_t)k_padded / QK8_1 * input_block_size;

  const block_q8_1 *x =
      (const block_q8_1 *)((const char *)all_inputs +
                           (size_t)current_batch * input_stride_bytes);
  const block_q_t *gate_w =
      (const block_q_t *)((const char *)gate_weights +
                          (size_t)expert_id * expert_stride_bytes);
  const block_q_t *up_w =
      (const block_q_t *)((const char *)up_weights +
                          (size_t)expert_id * expert_stride_bytes);

  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
  const int blocks_per_row_x = (int)blocks_per_row;

  float *out = all_outputs + (size_t)task_id * n;

  for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
    const int row = row0 + r;
    float g_sum = 0.0f;
    float u_sum = 0.0f;

    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (threadIdx.x % (qi / vdr));
      g_sum += vec_dot_q_cuda(&gate_w[kbx + (size_t)row * blocks_per_row],
                              &x[kby], kqs);
      u_sum += vec_dot_q_cuda(&up_w[kbx + (size_t)row * blocks_per_row],
                              &x[kby], kqs);
    }

    g_sum = warp_reduce_sum(g_sum);
    u_sum = warp_reduce_sum(u_sum);

    if (threadIdx.x == 0) {
      float activated =
          (act_type == 0) ? gelu_pytorch_tanh(g_sum) : silu(g_sum);
      out[row] = u_sum * activated;
    }
  }
}

// Fused down+aggregate kernel template
// Computes: output[batch] += topk_weight * down_proj(intermediate) for each
// expert Uses atomicAdd for cross-expert aggregation Grid: (ceildiv(n, NWARPS *
// ROWS_PER_WARP), topk, batch) Block: (WARP_SIZE, NWARPS, 1) Each warp
// independently handles ROWS_PER_WARP output rows
template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void moe_gemv_down_aggregate_impl(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices,
    const float *__restrict__ topk_weights_ptr, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded) {

  constexpr int ROWS_PER_WARP = 4;
  constexpr int NWARPS = 4;
  constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;

  const int warp_id = threadIdx.y;
  const int row0 = ROWS_PER_BLOCK * blockIdx.x + warp_id * ROWS_PER_WARP;
  const int current_topk = blockIdx.y;
  const int current_batch = blockIdx.z;

  if (row0 >= n)
    return;

  const int task_id = current_batch * topk + current_topk;
  const unsigned int expert_id = indices[task_id];
  const float tw = topk_weights_ptr[task_id];

  // Weight layout: [num_experts, n, k_blocks]
  const size_t weight_block_size = sizeof(block_q_t);
  const size_t input_block_size = sizeof(block_q8_1);
  const size_t blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t expert_stride_bytes =
      (size_t)n * blocks_per_row * weight_block_size;
  const size_t input_stride_bytes = (size_t)k_padded / QK8_1 * input_block_size;

  // Each topk slot has its own input row (input_dim1=topk for down proj)
  const block_q8_1 *x =
      (const block_q8_1 *)((const char *)all_inputs +
                           (size_t)task_id * input_stride_bytes);

  const block_q_t *w =
      (const block_q_t *)((const char *)all_weights +
                          (size_t)expert_id * expert_stride_bytes);

  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
  const int blocks_per_row_x = (int)blocks_per_row;

  // Output is aggregated across experts: [batch, n]
  float *out = all_outputs + (size_t)current_batch * n;

  for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
    const int row = row0 + r;
    float tmp = 0.0f;

    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (threadIdx.x % (qi / vdr));
      tmp +=
          vec_dot_q_cuda(&w[kbx + (size_t)row * blocks_per_row], &x[kby], kqs);
    }

    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
      atomicAdd(&out[row], tmp * tw);
    }
  }
}

// ============== Fused gate+up kernel instantiations ==============

#define FUSED_GATE_UP_KERNEL(suffix, qk_val, qi_val, block_type, vdr_val,      \
                             vd_fn)                                            \
  extern "C" __global__ void moe_gemv_fused_gate_up_##suffix(                  \
      const void *__restrict__ gate_weights,                                   \
      const void *__restrict__ up_weights,                                     \
      const void *__restrict__ all_inputs,                                     \
      const unsigned int *__restrict__ indices,                                \
      float *__restrict__ all_outputs, const int n, const int k,               \
      const int batch, const int topk, const int k_padded,                     \
      const int act_type) {                                                    \
    moe_gemv_fused_gate_up_impl<qk_val, qi_val, block_type, vdr_val, vd_fn>(   \
        gate_weights, up_weights, all_inputs, indices, all_outputs, n, k,      \
        batch, topk, k_padded, act_type);                                      \
  }

FUSED_GATE_UP_KERNEL(q8_0_q8_1, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                     vec_dot_q8_0_q8_1)
FUSED_GATE_UP_KERNEL(q4_0_q8_1, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                     vec_dot_q4_0_q8_1)
FUSED_GATE_UP_KERNEL(q4_1_q8_1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                     vec_dot_q4_1_q8_1)
FUSED_GATE_UP_KERNEL(q5_0_q8_1, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                     vec_dot_q5_0_q8_1)
FUSED_GATE_UP_KERNEL(q5_1_q8_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                     vec_dot_q5_1_q8_1)
FUSED_GATE_UP_KERNEL(q8_1_q8_1, QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                     vec_dot_q8_1_q8_1)
FUSED_GATE_UP_KERNEL(q2k_q8_1, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                     vec_dot_q2_K_q8_1)
FUSED_GATE_UP_KERNEL(q3k_q8_1, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                     vec_dot_q3_K_q8_1)
FUSED_GATE_UP_KERNEL(q4k_q8_1, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                     vec_dot_q4_K_q8_1)
FUSED_GATE_UP_KERNEL(q5k_q8_1, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                     vec_dot_q5_K_q8_1)
FUSED_GATE_UP_KERNEL(q6k_q8_1, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                     vec_dot_q6_K_q8_1)
FUSED_GATE_UP_KERNEL(iq4xs_q8_1, QK_K, 8, block_iq4_xs,
                     VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1)
FUSED_GATE_UP_KERNEL(iq4nl_q8_1, QK4_NL, 1, block_iq4_nl,
                     VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1)
FUSED_GATE_UP_KERNEL(q2_0_q8_1, QK2_0, 2, block_q2_0,
                     VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1)
FUSED_GATE_UP_KERNEL(iq3xxs_q8_1, QK_K, QI3_XXS, block_iq3_xxs,
                     VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1)
FUSED_GATE_UP_KERNEL(iq2s_q8_1, QK_K, QI2_S, block_iq2_s,
                     VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1)
FUSED_GATE_UP_KERNEL(iq3s_q8_1, QK_K, QI3_S, block_iq3_s,
                     VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1)

#define LORA_DECODE_KERNELS(suffix, qk_val, qi_val, block_type, vdr_val,       \
                            vd_fn)                                             \
  extern "C" __global__ void moe_gemv_gate_up_pair_##suffix(                   \
      const void *__restrict__ gate_weights,                                   \
      const void *__restrict__ up_weights,                                     \
      const void *__restrict__ all_inputs,                                     \
      const unsigned int *__restrict__ indices,                                \
      void *__restrict__ all_outputs, const int n, const int k,                \
      const int batch, const int topk, const int num_experts,                  \
      const int k_padded, const int output_type) {                             \
    moe_gemv_gate_up_pair_impl<qk_val, qi_val, block_type, vdr_val, vd_fn>(    \
        gate_weights, up_weights, all_inputs, indices, all_outputs, n, k,      \
        batch, topk, num_experts, k_padded, output_type);                      \
  }                                                                            \
  extern "C" __global__ void moe_gemv_lora_down_##suffix(                      \
      const void *__restrict__ all_weights,                                    \
      const void *__restrict__ all_inputs,                                     \
      const unsigned int *__restrict__ indices,                                \
      void *__restrict__ all_outputs, const int n, const int k,                \
      const int batch, const int topk, const int num_experts,                  \
      const int k_padded, const int output_type) {                             \
    moe_gemv_lora_down_impl<qk_val, qi_val, block_type, vdr_val, vd_fn>(       \
        all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,      \
        num_experts, k_padded, output_type);                                   \
  }

LORA_DECODE_KERNELS(q8_0_q8_1, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                    vec_dot_q8_0_q8_1)
LORA_DECODE_KERNELS(q4_0_q8_1, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                    vec_dot_q4_0_q8_1)
LORA_DECODE_KERNELS(q4_1_q8_1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                    vec_dot_q4_1_q8_1)
LORA_DECODE_KERNELS(q5_0_q8_1, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                    vec_dot_q5_0_q8_1)
LORA_DECODE_KERNELS(q5_1_q8_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                    vec_dot_q5_1_q8_1)
LORA_DECODE_KERNELS(q8_1_q8_1, QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                    vec_dot_q8_1_q8_1)
LORA_DECODE_KERNELS(q2k_q8_1, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                    vec_dot_q2_K_q8_1)
LORA_DECODE_KERNELS(q3k_q8_1, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                    vec_dot_q3_K_q8_1)
LORA_DECODE_KERNELS(q4k_q8_1, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                    vec_dot_q4_K_q8_1)
LORA_DECODE_KERNELS(q5k_q8_1, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                    vec_dot_q5_K_q8_1)
LORA_DECODE_KERNELS(q6k_q8_1, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                    vec_dot_q6_K_q8_1)

// ============== Fused down+aggregate kernel instantiations ==============

#define DOWN_AGGREGATE_KERNEL(suffix, qk_val, qi_val, block_type, vdr_val,     \
                              vd_fn)                                           \
  extern "C" __global__ void moe_gemv_down_aggregate_##suffix(                 \
      const void *__restrict__ all_weights,                                    \
      const void *__restrict__ all_inputs,                                     \
      const unsigned int *__restrict__ indices,                                \
      const float *__restrict__ topk_weights_ptr,                              \
      float *__restrict__ all_outputs, const int n, const int k,               \
      const int batch, const int topk, const int k_padded) {                   \
    moe_gemv_down_aggregate_impl<qk_val, qi_val, block_type, vdr_val, vd_fn>(  \
        all_weights, all_inputs, indices, topk_weights_ptr, all_outputs, n, k, \
        batch, topk, k_padded);                                                \
  }

DOWN_AGGREGATE_KERNEL(q8_0_q8_1, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                      vec_dot_q8_0_q8_1)
DOWN_AGGREGATE_KERNEL(q4_0_q8_1, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                      vec_dot_q4_0_q8_1)
DOWN_AGGREGATE_KERNEL(q4_1_q8_1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                      vec_dot_q4_1_q8_1)
DOWN_AGGREGATE_KERNEL(q5_0_q8_1, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                      vec_dot_q5_0_q8_1)
DOWN_AGGREGATE_KERNEL(q5_1_q8_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                      vec_dot_q5_1_q8_1)
DOWN_AGGREGATE_KERNEL(q8_1_q8_1, QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                      vec_dot_q8_1_q8_1)
DOWN_AGGREGATE_KERNEL(q2k_q8_1, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                      vec_dot_q2_K_q8_1)
DOWN_AGGREGATE_KERNEL(q3k_q8_1, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                      vec_dot_q3_K_q8_1)
DOWN_AGGREGATE_KERNEL(q4k_q8_1, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                      vec_dot_q4_K_q8_1)
DOWN_AGGREGATE_KERNEL(q5k_q8_1, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                      vec_dot_q5_K_q8_1)
DOWN_AGGREGATE_KERNEL(q6k_q8_1, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                      vec_dot_q6_K_q8_1)
DOWN_AGGREGATE_KERNEL(iq4xs_q8_1, QK_K, 8, block_iq4_xs,
                      VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1)
DOWN_AGGREGATE_KERNEL(iq4nl_q8_1, QK4_NL, 1, block_iq4_nl,
                      VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1)
DOWN_AGGREGATE_KERNEL(q2_0_q8_1, QK2_0, 2, block_q2_0,
                      VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1)
DOWN_AGGREGATE_KERNEL(iq3xxs_q8_1, QK_K, QI3_XXS, block_iq3_xxs,
                      VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1)
DOWN_AGGREGATE_KERNEL(iq2s_q8_1, QK_K, QI2_S, block_iq2_s,
                      VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1)
DOWN_AGGREGATE_KERNEL(iq3s_q8_1, QK_K, QI3_S, block_iq3_s,
                      VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1)

// ============== Fused gate+up launcher functions ==============

#define LAUNCH_FUSED_GATE_UP(suffix)                                           \
  extern "C" void launch_moe_gemv_fused_gate_up_##suffix(                      \
      const void *gate_weights, const void *up_weights,                        \
      const void *all_inputs, const unsigned int *indices, float *all_outputs, \
      int n, int k, int batch, int topk, int k_padded, int act_type,           \
      void *stream) {                                                          \
    const int NWARPS = 4;                                                      \
    const int ROWS_PER_WARP = 2;                                               \
    const int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;                         \
    dim3 grid(1 + (n - 1) / ROWS_PER_BLOCK, topk, batch);                      \
    dim3 block(WARP_SIZE, NWARPS, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    moe_gemv_fused_gate_up_##suffix<<<grid, block, 0, s>>>(                    \
        gate_weights, up_weights, all_inputs, indices, all_outputs, n, k,      \
        batch, topk, k_padded, act_type);                                      \
  }

LAUNCH_FUSED_GATE_UP(q8_0_q8_1)
LAUNCH_FUSED_GATE_UP(q4_0_q8_1)
LAUNCH_FUSED_GATE_UP(q4_1_q8_1)
LAUNCH_FUSED_GATE_UP(q5_0_q8_1)
LAUNCH_FUSED_GATE_UP(q5_1_q8_1)
LAUNCH_FUSED_GATE_UP(q8_1_q8_1)
LAUNCH_FUSED_GATE_UP(q2k_q8_1)
LAUNCH_FUSED_GATE_UP(q3k_q8_1)
LAUNCH_FUSED_GATE_UP(q4k_q8_1)
LAUNCH_FUSED_GATE_UP(q5k_q8_1)
LAUNCH_FUSED_GATE_UP(q6k_q8_1)
LAUNCH_FUSED_GATE_UP(iq4xs_q8_1)
LAUNCH_FUSED_GATE_UP(iq4nl_q8_1)
LAUNCH_FUSED_GATE_UP(q2_0_q8_1)
LAUNCH_FUSED_GATE_UP(iq3xxs_q8_1)
LAUNCH_FUSED_GATE_UP(iq2s_q8_1)
LAUNCH_FUSED_GATE_UP(iq3s_q8_1)

#define LAUNCH_LORA_DECODE(suffix)                                             \
  extern "C" int launch_moe_gemv_gate_up_pair_##suffix(                        \
      const void *gate_weights, const void *up_weights,                        \
      const void *all_inputs, const unsigned int *indices, void *all_outputs,  \
      int n, int k, int batch, int topk, int k_padded, int num_experts,        \
      int output_type, void *stream) {                                         \
    if (output_type < MOE_OUTPUT_F32 || output_type > MOE_OUTPUT_BF16)         \
      return static_cast<int>(cudaErrorInvalidValue);                          \
    constexpr int NWARPS = 4;                                                  \
    constexpr int ROWS_PER_WARP = 2;                                           \
    constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;                     \
    dim3 grid(1 + (n - 1) / ROWS_PER_BLOCK, topk, batch);                      \
    dim3 block(WARP_SIZE, NWARPS, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    moe_gemv_gate_up_pair_##suffix<<<grid, block, 0, s>>>(                     \
        gate_weights, up_weights, all_inputs, indices, all_outputs, n, k,      \
        batch, topk, num_experts, k_padded, output_type);                      \
    return static_cast<int>(cudaGetLastError());                               \
  }                                                                            \
  extern "C" int launch_moe_gemv_lora_down_##suffix(                           \
      const void *all_weights, const void *all_inputs,                         \
      const unsigned int *indices, void *all_outputs, int n, int k, int batch, \
      int topk, int k_padded, int num_experts, int output_type,                \
      void *stream) {                                                          \
    if (output_type < MOE_OUTPUT_F32 || output_type > MOE_OUTPUT_BF16)         \
      return static_cast<int>(cudaErrorInvalidValue);                          \
    constexpr int NWARPS = 4;                                                  \
    constexpr int ROWS_PER_WARP = 4;                                           \
    constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;                     \
    dim3 grid((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, topk, batch);         \
    dim3 block(WARP_SIZE, NWARPS, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    moe_gemv_lora_down_##suffix<<<grid, block, 0, s>>>(                        \
        all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,      \
        num_experts, k_padded, output_type);                                   \
    return static_cast<int>(cudaGetLastError());                               \
  }

LAUNCH_LORA_DECODE(q8_0_q8_1)
LAUNCH_LORA_DECODE(q4_0_q8_1)
LAUNCH_LORA_DECODE(q4_1_q8_1)
LAUNCH_LORA_DECODE(q5_0_q8_1)
LAUNCH_LORA_DECODE(q5_1_q8_1)
LAUNCH_LORA_DECODE(q8_1_q8_1)
LAUNCH_LORA_DECODE(q2k_q8_1)
LAUNCH_LORA_DECODE(q3k_q8_1)
LAUNCH_LORA_DECODE(q4k_q8_1)
LAUNCH_LORA_DECODE(q5k_q8_1)
LAUNCH_LORA_DECODE(q6k_q8_1)

// ============== Fused down+aggregate launcher functions ==============

#define LAUNCH_DOWN_AGGREGATE(suffix)                                          \
  extern "C" void launch_moe_gemv_down_aggregate_##suffix(                     \
      const void *all_weights, const void *all_inputs,                         \
      const unsigned int *indices, const float *topk_weights_ptr,              \
      float *all_outputs, int n, int k, int batch, int topk, int k_padded,     \
      void *stream) {                                                          \
    const int NWARPS = 4;                                                      \
    const int ROWS_PER_WARP = 4;                                               \
    const int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;                         \
    dim3 grid((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, topk, batch);         \
    dim3 block(WARP_SIZE, NWARPS, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    moe_gemv_down_aggregate_##suffix<<<grid, block, 0, s>>>(                   \
        all_weights, all_inputs, indices, topk_weights_ptr, all_outputs, n, k, \
        batch, topk, k_padded);                                                \
  }

LAUNCH_DOWN_AGGREGATE(q8_0_q8_1)
LAUNCH_DOWN_AGGREGATE(q4_0_q8_1)
LAUNCH_DOWN_AGGREGATE(q4_1_q8_1)
LAUNCH_DOWN_AGGREGATE(q5_0_q8_1)
LAUNCH_DOWN_AGGREGATE(q5_1_q8_1)
LAUNCH_DOWN_AGGREGATE(q8_1_q8_1)
LAUNCH_DOWN_AGGREGATE(q2k_q8_1)
LAUNCH_DOWN_AGGREGATE(q3k_q8_1)
LAUNCH_DOWN_AGGREGATE(q4k_q8_1)
LAUNCH_DOWN_AGGREGATE(q5k_q8_1)
LAUNCH_DOWN_AGGREGATE(q6k_q8_1)
LAUNCH_DOWN_AGGREGATE(iq4xs_q8_1)
LAUNCH_DOWN_AGGREGATE(iq4nl_q8_1)
LAUNCH_DOWN_AGGREGATE(q2_0_q8_1)
LAUNCH_DOWN_AGGREGATE(iq3xxs_q8_1)
LAUNCH_DOWN_AGGREGATE(iq2s_q8_1)
LAUNCH_DOWN_AGGREGATE(iq3s_q8_1)
