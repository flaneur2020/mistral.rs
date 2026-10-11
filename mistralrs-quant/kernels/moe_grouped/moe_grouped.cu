// Grouped MoE GEMM kernel for prefill optimization.
//
// Instead of launching one block per (output_feature, token, topk_slot) like
// the indexed_moe kernel, this groups tokens by expert and iterates over them
// within each block. This reduces block count by ~500x and enables weight
// data reuse across tokens via L1 cache.
//
// Grid: (N, num_experts) vs indexed_moe's (N, batch, topk)
// For Gemma4 8K tokens: 90K blocks vs 44.8M blocks.

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#include "../iq2s_grid.cuh"
#include "../iq3s_grid.cuh"

#define WARP_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256

// ============== Helper functions ==============

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, 32);
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

// The 16-entry non-linear IQ4_XS/IQ4_NL codebook, and the 4-bit table lookup.
static __device__ __align__(16) const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

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

// ============== Block type definitions ==============

#define QK_K 256
#define K_SCALE_SIZE 12

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

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
  half d;
  uint8_t qs[QK4_0 / 2];
} block_q4_0;

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
  half2 dm;
  uint8_t qs[QK4_1 / 2];
} block_q4_1;

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
  half d;
  uint8_t qh[4];
  uint8_t qs[QK5_0 / 2];
} block_q5_0;

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
  half2 dm;
  uint8_t qh[4];
  uint8_t qs[QK5_1 / 2];
} block_q5_1;

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

// IQ4_XS: eight 32-element groups, `qi` = groups per super-block.
#define QI4_XS 8
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




// VDR constants (same as indexed_moe)
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

// ============== vec_dot implementations ==============

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
  return sumi * dm4f.x * ds8f.x + dm4f.y * ds8f.y / (QI8_1 / (vdr * QR4_1));
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
  return sumi * dm5f.x * ds8f.x + dm5f.y * ds8f.y / (QI5_1 / vdr);
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

// ============== vec_dot wrapper functions ==============

typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ vbq,
                                  const block_q8_1 *__restrict__ bq8_1,
                                  const int &iqs);

static __device__ __forceinline__ float
vd_q8_0_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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
vd_q4_0_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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
vd_q4_1_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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
vd_q5_0_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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
vd_q5_1_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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
vd_q8_1_q8_1(const void *__restrict__ vbq,
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
vd_q2_K_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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
vd_q3_K_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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
vd_q4_K_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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

static __device__ __forceinline__ float
vd_q5_K_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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
vd_q6_K_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
             const int &iqs) {
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

// IQ4_XS: one 32-element group per call, `iqs` = group index (0..7).
static __device__ __forceinline__ float
vd_iq4xs_q8_1(const void *__restrict__ vbq,
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
  const int ls = ((bq4->scales_l[iqs / 2] >> (4 * (iqs % 2))) & 0x0f) |
                 (((bq4->scales_h >> (2 * iqs)) & 0x03) << 4);
  sumi *= ls - 32;
  const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs].ds);
  return d * sumi;
}

// IQ4_NL: the whole 32-element block per call (qi = vdr = 1), `iqs` is 0.
static __device__ __forceinline__ float
vd_iq4nl_q8_1(const void *__restrict__ vbq,
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
vd_q2_0_q8_1(const void *__restrict__ vbq,
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

// ============== Dispatch kernels ==============

// Phase 1: Count tokens per expert using atomics
static __global__ void moe_dispatch_count_kernel(
    const int32_t *__restrict__ topk_ids, // [total_assignments] flattened
    int32_t *__restrict__ expert_counts,  // [num_experts] zero-initialized
    const int total_assignments) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_assignments) {
    const int expert = topk_ids[idx];
    atomicAdd(&expert_counts[expert], 1);
  }
}

// Phase 2: Exclusive prefix sum for expert_bounds (single block, <=1024
// experts)
static __global__ void moe_dispatch_prefix_sum_kernel(
    const int32_t *__restrict__ expert_counts, // [num_experts]
    int32_t *__restrict__ expert_bounds,       // [num_experts + 1]
    const int num_experts) {
  // Simple sequential prefix sum - fine for num_experts <= 1024
  // Only one thread does this work
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    expert_bounds[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
      expert_bounds[i + 1] = expert_bounds[i] + expert_counts[i];
    }
  }
}

// Phase 3: Scatter token indices into sorted positions
static __global__ void moe_dispatch_scatter_kernel(
    const int32_t *__restrict__ topk_ids, // [total_assignments] flattened
    int32_t
        *__restrict__ expert_cursors, // [num_experts] init to expert_bounds[i]
    int32_t *__restrict__ sorted_token_ids,  // [total_assignments] output
    int32_t *__restrict__ sorted_source_ids, // [total_assignments] output
    const int total_assignments, const int topk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_assignments) {
    const int expert = topk_ids[idx];
    const int pos = atomicAdd(&expert_cursors[expert], 1);
    sorted_token_ids[pos] = idx; // flat index into topk_ids: token = idx/topk
    if (sorted_source_ids) {
      sorted_source_ids[pos] = idx / topk;
    }
  }
}

template <typename InputT, typename OutputT>
static __global__ void moe_weighted_reduce_flat_kernel(
    const InputT *__restrict__ inputs, const float *__restrict__ topk_weights,
    OutputT *__restrict__ outputs, const int num_tokens, const int hidden,
    const int topk) {
  const int token = blockIdx.x;
  const int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= num_tokens)
    return;

  extern __shared__ float weights[];
  for (int slot = threadIdx.x; slot < topk; slot += blockDim.x) {
    weights[slot] = topk_weights[token * topk + slot];
  }
  __syncthreads();

  if (h >= hidden)
    return;

  const size_t input_base = (size_t)token * topk * hidden + h;
  float acc = 0.0f;
  for (int slot = 0; slot < topk; ++slot) {
    acc += (float)inputs[input_base + (size_t)slot * hidden] * weights[slot];
  }
  outputs[(size_t)token * hidden + h] = (OutputT)acc;
}

// ============== Tiled MoE GEMM kernel (all quant types) ==============
//
// Each block processes MMQ_Y output features x MMQ_X tokens for one expert.
// Both weight blocks and input Q8_1 blocks are loaded into shared memory
// as raw bytes, then vec_dot is called on the shared memory pointers.
// This works for ANY quant type since vec_dot handles unpacking internally.
//
// Grid: (ceil(N/MMQ_Y), num_experts), Block: (32, MMQ_NWARPS)
// K processed in chunks of K_TILE_BLOCKS weight blocks at a time.

#define MMQ_X 64
#define MMQ_Y 64
#define MMQ_NWARPS 8

// K_TILE_BLOCKS controls how many weight blocks are processed per K-tile.
// For K-type quants (QK_K=256), each weight block maps to 8 Q8_1 input blocks,
// so shared memory scales as K_TILE_BLOCKS * 8.  Use a smaller tile for K-types
// to keep shared memory under the 48 KB default CUDA limit.
#define K_TILE_BLOCKS_SMALL 1 // for QK_K=256 types (Q2K..Q6K)
#define K_TILE_BLOCKS_LARGE 8 // for QK=32 types (Q4_0..Q8_1)

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int k_tile_blocks>
static __device__ void moe_tiled_gemm_impl(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const int32_t *__restrict__ expert_bounds,
    const int32_t *__restrict__ sorted_token_ids,
    const float *__restrict__ topk_weights, float *__restrict__ all_outputs,
    const int N, const int K, const int K_padded, const int num_experts,
    const int topk, const int input_dim1) {

  const int n_base = blockIdx.x * MMQ_Y;
  const int expert = blockIdx.y;
  if (n_base >= N || expert >= num_experts)
    return;

  const int t_start = expert_bounds[expert];
  const int t_end = expert_bounds[expert + 1];
  if (t_start >= t_end)
    return;

  const int blocks_per_row_w = K / qk;
  const int blocks_per_row_x = K_padded / QK8_1;
  constexpr int x_blocks_per_w_block =
      qk / QK8_1; // Q8_1 blocks per weight block
  constexpr int x_blocks_per_k_tile = k_tile_blocks * x_blocks_per_w_block;
  constexpr size_t w_block_bytes = sizeof(block_q_t);
  constexpr size_t x_block_bytes = sizeof(block_q8_1);

  const char *w_bytes = (const char *)all_weights;
  const char *x_bytes = (const char *)all_inputs;
  const size_t expert_w_off =
      (size_t)expert * N * blocks_per_row_w * w_block_bytes;

  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = MMQ_NWARPS * WARP_SIZE;

  // Shared memory: w_tile + x_tile + tok_cache
  extern __shared__ char smem[];
  char *w_tile = smem;
  constexpr size_t w_tile_bytes = MMQ_Y * k_tile_blocks * w_block_bytes;
  char *x_tile = w_tile + w_tile_bytes;
  constexpr size_t x_tile_bytes = MMQ_X * x_blocks_per_k_tile * x_block_bytes;
  int *tok_cache = (int *)(x_tile + x_tile_bytes);

  constexpr int acc_rows = MMQ_Y / WARP_SIZE;
  constexpr int acc_cols = MMQ_X / MMQ_NWARPS;

  for (int t_base = t_start; t_base < t_end; t_base += MMQ_X) {
    const int m_count = min(MMQ_X, t_end - t_base);

    __syncthreads();
    for (int idx = tid; idx < m_count; idx += nthreads) {
      const int ti = t_base + idx;
      const int flat_idx = sorted_token_ids[ti];
      tok_cache[idx] = (input_dim1 == 0)
                           ? ti
                           : ((input_dim1 == 1) ? (flat_idx / topk) : flat_idx);
    }

    float acc[acc_cols][acc_rows];
#pragma unroll
    for (int j = 0; j < acc_cols; j++)
#pragma unroll
      for (int i = 0; i < acc_rows; i++)
        acc[j][i] = 0.0f;

    for (int kb_base = 0; kb_base < blocks_per_row_w;
         kb_base += k_tile_blocks) {
      const int kb_count = min(k_tile_blocks, blocks_per_row_w - kb_base);
      const int xb_base = kb_base * x_blocks_per_w_block;

      __syncthreads();

      // Load weight tile: MMQ_Y rows x kb_count blocks (byte-level copy)
      {
        const int total_bytes = MMQ_Y * kb_count * w_block_bytes;
        const int row_bytes = kb_count * w_block_bytes;
        const int row_bytes_padded = k_tile_blocks * w_block_bytes;
        for (int idx = tid; idx < total_bytes; idx += nthreads) {
          const int row = idx / row_bytes;
          const int col = idx % row_bytes;
          const int n_row = n_base + row;
          const int dst = row * row_bytes_padded + col;
          if (n_row < N) {
            const size_t src_off =
                expert_w_off +
                (size_t)n_row * blocks_per_row_w * w_block_bytes +
                (size_t)kb_base * w_block_bytes + col;
            w_tile[dst] = w_bytes[src_off];
          } else {
            w_tile[dst] = 0;
          }
        }
      }

      // Load input tile: m_count rows x xb_count Q8_1 blocks (byte-level copy)
      {
        const int xb_count = kb_count * x_blocks_per_w_block;
        const int total_bytes = m_count * xb_count * x_block_bytes;
        const int row_bytes = xb_count * x_block_bytes;
        const int row_bytes_padded = x_blocks_per_k_tile * x_block_bytes;
        for (int idx = tid; idx < total_bytes; idx += nthreads) {
          const int row = idx / row_bytes;
          const int col = idx % row_bytes;
          const int dst = row * row_bytes_padded + col;
          if (row < m_count) {
            const int input_row = tok_cache[row];
            const size_t src_off =
                (size_t)input_row * blocks_per_row_x * x_block_bytes +
                (size_t)xb_base * x_block_bytes + col;
            x_tile[dst] = x_bytes[src_off];
          } else {
            x_tile[dst] = 0;
          }
        }
      }

      __syncthreads();

      // Compute: vec_dot on shared memory block pointers
      for (int kb = 0; kb < kb_count; kb++) {
#pragma unroll
        for (int i_local = 0; i_local < acc_rows; i_local++) {
          const int i = i_local * WARP_SIZE + threadIdx.x;
          if (n_base + i >= N)
            continue;

          const block_q_t *w_blk =
              (const block_q_t *)(w_tile +
                                  (size_t)i * k_tile_blocks * w_block_bytes +
                                  (size_t)kb * w_block_bytes);

#pragma unroll
          for (int j_local = 0; j_local < acc_cols; j_local++) {
            const int j = threadIdx.y * acc_cols + j_local;
            if (j >= m_count)
              continue;

            const block_q8_1 *x_blk =
                (const block_q8_1 *)(x_tile +
                                     (size_t)j * x_blocks_per_k_tile *
                                         x_block_bytes +
                                     (size_t)kb * x_blocks_per_w_block *
                                         x_block_bytes);

            // Sum over all sub-elements in this block using vec_dot
            for (int iqs = 0; iqs < qi; iqs += vdr) {
              acc[j_local][i_local] += vec_dot_q_cuda(w_blk, x_blk, iqs);
            }
          }
        }
      }
    } // end K loop

// Write results
#pragma unroll
    for (int j_local = 0; j_local < acc_cols; j_local++) {
      const int j = threadIdx.y * acc_cols + j_local;
      if (j >= m_count)
        continue;
      const int ti = t_base + j;
      const int flat_idx = sorted_token_ids[ti];
#pragma unroll
      for (int i_local = 0; i_local < acc_rows; i_local++) {
        const int n_row = i_local * WARP_SIZE + threadIdx.x + n_base;
        if (n_row >= N)
          continue;
        if (topk_weights) {
          atomicAdd(&all_outputs[(size_t)(flat_idx / topk) * N + n_row],
                    acc[j_local][i_local] * topk_weights[flat_idx]);
        } else {
          all_outputs[(size_t)ti * N + n_row] = acc[j_local][i_local];
        }
      }
    }
  }
}

// ============== Kernel instantiations ==============

#define MOE_KERNEL_INST(suffix, QK, QI, BLOCK_T, VDR, VEC_DOT, KTB)            \
  extern "C" __global__ void moe_grouped_gemm_##suffix(                        \
      const void *all_weights, const void *all_inputs,                         \
      const int32_t *expert_bounds, const int32_t *sorted_token_ids,           \
      const float *topk_weights, float *all_outputs, int N, int K,             \
      int K_padded, int num_experts, int topk, int input_dim1) {               \
    moe_tiled_gemm_impl<QK, QI, BLOCK_T, VDR, VEC_DOT, KTB>(                   \
        all_weights, all_inputs, expert_bounds, sorted_token_ids,              \
        topk_weights, all_outputs, N, K, K_padded, num_experts, topk,          \
        input_dim1);                                                           \
  }

// Q8_0 uses the explicit dp4a shared-memory tiled kernel for maximum
// performance. The generic tiled kernel is used for all other quant types.
#define K_TILE 256
#define K_TILE_Q8 (K_TILE / QK8_0)
#define W_QS_STRIDE (K_TILE / 4)
#define W_QS_SIZE (MMQ_Y * W_QS_STRIDE)
#define W_D_SIZE (MMQ_Y * K_TILE_Q8)
#define X_QS_STRIDE (K_TILE / 4)
#define X_QS_SIZE (MMQ_X * X_QS_STRIDE)
#define X_D_SIZE (MMQ_X * K_TILE_Q8)

static __device__ void moe_tiled_gemm_q8_0_impl(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const int32_t *__restrict__ expert_bounds,
    const int32_t *__restrict__ sorted_token_ids,
    const float *__restrict__ topk_weights, float *__restrict__ all_outputs,
    const int N, const int K, const int K_padded, const int num_experts,
    const int topk, const int input_dim1) {
  const int n_base = blockIdx.x * MMQ_Y;
  const int expert = blockIdx.y;
  if (n_base >= N || expert >= num_experts)
    return;
  const int t_start = expert_bounds[expert];
  const int t_end = expert_bounds[expert + 1];
  if (t_start >= t_end)
    return;
  const int blocks_per_row_w = K / QK8_0;
  const int blocks_per_row_x = K_padded / QK8_1;
  const block_q8_0 *w_all = (const block_q8_0 *)all_weights;
  const block_q8_1 *x_all = (const block_q8_1 *)all_inputs;
  const size_t expert_w_offset = (size_t)expert * N * blocks_per_row_w;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = MMQ_NWARPS * WARP_SIZE;
  extern __shared__ char smem_q8[];
  int *w_qs = (int *)smem_q8;
  float *w_d = (float *)(w_qs + W_QS_SIZE);
  int *x_qs = (int *)(w_d + W_D_SIZE);
  float *x_d = (float *)(x_qs + X_QS_SIZE);
  int *tok_cache = (int *)(x_d + X_D_SIZE);
  constexpr int acc_rows = MMQ_Y / WARP_SIZE;
  constexpr int acc_cols = MMQ_X / MMQ_NWARPS;
  for (int t_base = t_start; t_base < t_end; t_base += MMQ_X) {
    const int m_count = min(MMQ_X, t_end - t_base);
    __syncthreads();
    for (int idx = tid; idx < m_count; idx += nthreads) {
      const int ti = t_base + idx;
      const int flat_idx = sorted_token_ids[ti];
      tok_cache[idx] = (input_dim1 == 0)
                           ? ti
                           : ((input_dim1 == 1) ? (flat_idx / topk) : flat_idx);
    }
    float acc[acc_cols][acc_rows];
    for (int j = 0; j < acc_cols; j++)
      for (int i = 0; i < acc_rows; i++)
        acc[j][i] = 0.0f;
    for (int k_base = 0; k_base < K; k_base += K_TILE) {
      const int k_tile_actual = min(K_TILE, K - k_base);
      const int k_tile_q8 = k_tile_actual / QK8_0;
      __syncthreads();
      for (int idx = tid; idx < W_QS_SIZE; idx += nthreads) {
        const int row = idx / W_QS_STRIDE, col = idx % W_QS_STRIDE;
        const int n_row = n_base + row, k_elem = col * 4;
        if (n_row < N && k_elem < k_tile_actual) {
          const int qb = (k_base + k_elem) / QK8_0, qo = (k_elem % QK8_0) / 4;
          w_qs[idx] = get_int_from_int8(
              w_all[expert_w_offset + (size_t)n_row * blocks_per_row_w + qb].qs,
              qo);
        } else
          w_qs[idx] = 0;
      }
      for (int idx = tid; idx < MMQ_Y * k_tile_q8; idx += nthreads) {
        const int row = idx / k_tile_q8, kb = idx % k_tile_q8;
        const int n_row = n_base + row, qb = k_base / QK8_0 + kb;
        w_d[row * K_TILE_Q8 + kb] =
            (n_row < N && qb < blocks_per_row_w)
                ? __half2float(w_all[expert_w_offset +
                                     (size_t)n_row * blocks_per_row_w + qb]
                                   .d)
                : 0.0f;
      }
      for (int idx = tid; idx < X_QS_SIZE; idx += nthreads) {
        const int row = idx / X_QS_STRIDE, col = idx % X_QS_STRIDE,
                  k_elem = col * 4;
        if (row < m_count && k_elem < k_tile_actual) {
          const int input_row = tok_cache[row];
          const int qb = (k_base + k_elem) / QK8_1, qo = (k_elem % QK8_1) / 4;
          x_qs[idx] = get_int_from_int8_aligned(
              x_all[(size_t)input_row * blocks_per_row_x + qb].qs, qo);
        } else
          x_qs[idx] = 0;
      }
      for (int idx = tid; idx < MMQ_X * k_tile_q8; idx += nthreads) {
        const int row = idx / k_tile_q8, kb = idx % k_tile_q8;
        if (row < m_count) {
          const int input_row = tok_cache[row], qb = k_base / QK8_1 + kb;
          x_d[row * K_TILE_Q8 + kb] =
              (qb < blocks_per_row_x)
                  ? __low2float(
                        x_all[(size_t)input_row * blocks_per_row_x + qb].ds)
                  : 0.0f;
        }
      }
      __syncthreads();
#pragma unroll
      for (int kb = 0; kb < K_TILE_Q8; kb++) {
        const int k_start = kb * (QK8_0 / 4);
#pragma unroll
        for (int i_local = 0; i_local < acc_rows; i_local++) {
          const int i = i_local * WARP_SIZE + threadIdx.x;
          if (n_base + i >= N)
            continue;
          const float d_w = w_d[i * K_TILE_Q8 + kb];
          int w_vals[8];
#pragma unroll
          for (int kk = 0; kk < 8; kk++)
            w_vals[kk] = w_qs[i * W_QS_STRIDE + k_start + kk];
#pragma unroll
          for (int j_local = 0; j_local < acc_cols; j_local++) {
            const int j = threadIdx.y * acc_cols + j_local;
            if (j >= m_count)
              continue;
            const float d_x = x_d[j * K_TILE_Q8 + kb];
            int sumi = 0;
#pragma unroll
            for (int kk = 0; kk < 8; kk++)
              sumi = ggml_cuda_dp4a(w_vals[kk],
                                    x_qs[j * X_QS_STRIDE + k_start + kk], sumi);
            acc[j_local][i_local] += d_w * d_x * (float)sumi;
          }
        }
      }
    }
#pragma unroll
    for (int j_local = 0; j_local < acc_cols; j_local++) {
      const int j = threadIdx.y * acc_cols + j_local;
      if (j >= m_count)
        continue;
      const int ti = t_base + j, flat_idx = sorted_token_ids[ti];
#pragma unroll
      for (int i_local = 0; i_local < acc_rows; i_local++) {
        const int n_row = i_local * WARP_SIZE + threadIdx.x + n_base;
        if (n_row >= N)
          continue;
        if (topk_weights)
          atomicAdd(&all_outputs[(size_t)(flat_idx / topk) * N + n_row],
                    acc[j_local][i_local] * topk_weights[flat_idx]);
        else
          all_outputs[(size_t)ti * N + n_row] = acc[j_local][i_local];
      }
    }
  }
}

extern "C" __global__ void moe_grouped_gemm_q8_0(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs, int N, int K, int K_padded,
    int num_experts, int topk, int input_dim1) {
  moe_tiled_gemm_q8_0_impl(all_weights, all_inputs, expert_bounds,
                           sorted_token_ids, topk_weights, all_outputs, N, K,
                           K_padded, num_experts, topk, input_dim1);
}
// Other quant types use the generic tiled kernel:

MOE_KERNEL_INST(q4_0, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                vd_q4_0_q8_1, K_TILE_BLOCKS_LARGE)
MOE_KERNEL_INST(q4_1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                vd_q4_1_q8_1, K_TILE_BLOCKS_LARGE)
MOE_KERNEL_INST(q5_0, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                vd_q5_0_q8_1, K_TILE_BLOCKS_LARGE)
MOE_KERNEL_INST(q5_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                vd_q5_1_q8_1, K_TILE_BLOCKS_LARGE)
MOE_KERNEL_INST(q8_1, QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                vd_q8_1_q8_1, K_TILE_BLOCKS_LARGE)
MOE_KERNEL_INST(q2k, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vd_q2_K_q8_1,
                K_TILE_BLOCKS_SMALL)
MOE_KERNEL_INST(q3k, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vd_q3_K_q8_1,
                K_TILE_BLOCKS_SMALL)
MOE_KERNEL_INST(q4k, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vd_q4_K_q8_1,
                K_TILE_BLOCKS_SMALL)
MOE_KERNEL_INST(q5k, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vd_q5_K_q8_1,
                K_TILE_BLOCKS_SMALL)
MOE_KERNEL_INST(q6k, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vd_q6_K_q8_1,
                K_TILE_BLOCKS_SMALL)
MOE_KERNEL_INST(iq4xs, QK_K, QI4_XS, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ,
                vd_iq4xs_q8_1, K_TILE_BLOCKS_SMALL)
MOE_KERNEL_INST(iq4nl, QK4_NL, 1, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ,
                vd_iq4nl_q8_1, K_TILE_BLOCKS_LARGE)
MOE_KERNEL_INST(q2_0, QK2_0, 2, block_q2_0, VDR_Q2_0_Q8_1_MMVQ,
                vd_q2_0_q8_1, 4)
MOE_KERNEL_INST(iq3xxs, QK_K, QI3_XXS, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ,
                vec_dot_iq3_xxs_q8_1, K_TILE_BLOCKS_SMALL)
MOE_KERNEL_INST(iq2s, QK_K, QI2_S, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ,
                vec_dot_iq2_s_q8_1, K_TILE_BLOCKS_SMALL)
MOE_KERNEL_INST(iq3s, QK_K, QI3_S, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ,
                vec_dot_iq3_s_q8_1, K_TILE_BLOCKS_SMALL)

// ============== C wrapper functions for FFI ==============

extern "C" void launch_moe_dispatch(
    const int32_t *topk_ids, int32_t *expert_bounds, int32_t *sorted_token_ids,
    int32_t *sorted_source_ids, int total_assignments, int num_experts,
    int topk, int32_t *expert_counts, int32_t *expert_cursors, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);

  cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), s);
  {
    int threads = 256;
    int blocks = (total_assignments + threads - 1) / threads;
    moe_dispatch_count_kernel<<<blocks, threads, 0, s>>>(
        topk_ids, expert_counts, total_assignments);
  }

  moe_dispatch_prefix_sum_kernel<<<1, 1, 0, s>>>(expert_counts, expert_bounds,
                                                 num_experts);

  cudaMemcpyAsync(expert_cursors, expert_bounds, num_experts * sizeof(int32_t),
                  cudaMemcpyDeviceToDevice, s);

  {
    int threads = 256;
    int blocks = (total_assignments + threads - 1) / threads;
    moe_dispatch_scatter_kernel<<<blocks, threads, 0, s>>>(
        topk_ids, expert_cursors, sorted_token_ids, sorted_source_ids,
        total_assignments, topk);
  }
}

extern "C" int launch_moe_weighted_reduce_flat(const float *inputs,
                                               const float *topk_weights,
                                               float *outputs, int num_tokens,
                                               int hidden, int topk,
                                               void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int threads = 256;
  dim3 grid(num_tokens, 1 + (hidden - 1) / threads);
  moe_weighted_reduce_flat_kernel<<<grid, threads, topk * sizeof(float), s>>>(
      inputs, topk_weights, outputs, num_tokens, hidden, topk);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_moe_weighted_reduce_flat_bf16(const float *inputs,
                                                    const float *topk_weights,
                                                    __nv_bfloat16 *outputs,
                                                    int num_tokens, int hidden,
                                                    int topk, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int threads = 256;
  dim3 grid(num_tokens, 1 + (hidden - 1) / threads);
  moe_weighted_reduce_flat_kernel<<<grid, threads, topk * sizeof(float), s>>>(
      inputs, topk_weights, outputs, num_tokens, hidden, topk);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_moe_weighted_reduce_flat_f16_input(
    const half *inputs, const float *topk_weights, half *outputs,
    int num_tokens, int hidden, int topk, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int threads = 256;
  dim3 grid(num_tokens, 1 + (hidden - 1) / threads);
  moe_weighted_reduce_flat_kernel<<<grid, threads, topk * sizeof(float), s>>>(
      inputs, topk_weights, outputs, num_tokens, hidden, topk);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_moe_weighted_reduce_flat_bf16_input(
    const __nv_bfloat16 *inputs, const float *topk_weights,
    __nv_bfloat16 *outputs, int num_tokens, int hidden, int topk,
    void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int threads = 256;
  dim3 grid(num_tokens, 1 + (hidden - 1) / threads);
  moe_weighted_reduce_flat_kernel<<<grid, threads, topk * sizeof(float), s>>>(
      inputs, topk_weights, outputs, num_tokens, hidden, topk);
  return static_cast<int>(cudaGetLastError());
}

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// Compute shared memory size for a given quant type and tile configuration
template <typename block_q_t, int qk, int ktb>
static size_t moe_tiled_smem_bytes() {
  constexpr int x_blocks_per_k_tile = ktb * (qk / QK8_1);
  constexpr size_t w_bytes = MMQ_Y * ktb * sizeof(block_q_t);
  constexpr size_t x_bytes = MMQ_X * x_blocks_per_k_tile * sizeof(block_q8_1);
  return w_bytes + x_bytes + MMQ_X * sizeof(int);
}

#define LAUNCH_MOE_TILED_GEMM(suffix, BLOCK_T, QK_VAL, KTB)                    \
  extern "C" void launch_moe_grouped_gemm_##suffix(                            \
      const void *all_weights, const void *all_inputs,                         \
      const int32_t *expert_bounds, const int32_t *sorted_token_ids,           \
      const float *topk_weights, float *all_outputs, int N, int K,             \
      int K_padded, int num_experts, int topk, int input_dim1, void *stream) { \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    dim3 grid(CEILDIV(N, MMQ_Y), num_experts);                                 \
    dim3 block(WARP_SIZE, MMQ_NWARPS, 1);                                      \
    size_t smem = moe_tiled_smem_bytes<BLOCK_T, QK_VAL, KTB>();                \
    moe_grouped_gemm_##suffix<<<grid, block, smem, s>>>(                       \
        all_weights, all_inputs, expert_bounds, sorted_token_ids,              \
        topk_weights, all_outputs, N, K, K_padded, num_experts, topk,          \
        input_dim1);                                                           \
  }

// Q8_0 launch uses the dp4a-specific smem layout
extern "C" void launch_moe_grouped_gemm_q8_0(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs, int N, int K, int K_padded,
    int num_experts, int topk, int input_dim1, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  dim3 grid(CEILDIV(N, MMQ_Y), num_experts);
  dim3 block(WARP_SIZE, MMQ_NWARPS, 1);
  size_t smem = (W_QS_SIZE + X_QS_SIZE) * sizeof(int) +
                (W_D_SIZE + X_D_SIZE) * sizeof(float) + MMQ_X * sizeof(int);
  moe_grouped_gemm_q8_0<<<grid, block, smem, s>>>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids, topk_weights,
      all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}
// Other quant types:
LAUNCH_MOE_TILED_GEMM(q4_0, block_q4_0, QK4_0, K_TILE_BLOCKS_LARGE)
LAUNCH_MOE_TILED_GEMM(q4_1, block_q4_1, QK4_1, K_TILE_BLOCKS_LARGE)
LAUNCH_MOE_TILED_GEMM(q5_0, block_q5_0, QK5_0, K_TILE_BLOCKS_LARGE)
LAUNCH_MOE_TILED_GEMM(q5_1, block_q5_1, QK5_1, K_TILE_BLOCKS_LARGE)
LAUNCH_MOE_TILED_GEMM(q8_1, block_q8_1, QK8_1, K_TILE_BLOCKS_LARGE)
LAUNCH_MOE_TILED_GEMM(q2k, block_q2_K, QK_K, K_TILE_BLOCKS_SMALL)
LAUNCH_MOE_TILED_GEMM(q3k, block_q3_K, QK_K, K_TILE_BLOCKS_SMALL)
LAUNCH_MOE_TILED_GEMM(q4k, block_q4_K, QK_K, K_TILE_BLOCKS_SMALL)
LAUNCH_MOE_TILED_GEMM(q5k, block_q5_K, QK_K, K_TILE_BLOCKS_SMALL)
LAUNCH_MOE_TILED_GEMM(q6k, block_q6_K, QK_K, K_TILE_BLOCKS_SMALL)
LAUNCH_MOE_TILED_GEMM(iq4xs, block_iq4_xs, QK_K, K_TILE_BLOCKS_SMALL)
LAUNCH_MOE_TILED_GEMM(iq4nl, block_iq4_nl, QK4_NL, K_TILE_BLOCKS_LARGE)
LAUNCH_MOE_TILED_GEMM(q2_0, block_q2_0, QK2_0, 4)
LAUNCH_MOE_TILED_GEMM(iq3xxs, block_iq3_xxs, QK_K, K_TILE_BLOCKS_SMALL)
LAUNCH_MOE_TILED_GEMM(iq2s, block_iq2_s, QK_K, K_TILE_BLOCKS_SMALL)
LAUNCH_MOE_TILED_GEMM(iq3s, block_iq3_s, QK_K, K_TILE_BLOCKS_SMALL)
