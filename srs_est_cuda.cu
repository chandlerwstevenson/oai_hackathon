/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/* GPU SRS channel estimator: the module the gNB loads with --loader.srs_est.shlibversion _cuda.
 *
 * This is the skeleton to fill in: it is loaded, initialized and called exactly like a finished
 * module, but srs_est_run() does not estimate anything yet and reports a failure, so the gNB falls
 * back to its fixed-point estimator for every SRS (and exports nothing). See srs_est_interface.h
 * for what goes in and what has to come out, doc/isac_srs_demo.md for the task and how to measure
 * the result. */

#include <cuda_runtime.h> 
#include <cufft.h>   
#include <stdio.h>

extern "C" {
#include "srs_est_interface.h"
}

#define CK(x)                                                                                     \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) {                                                                      \
      fprintf(stderr, "srs_est_cuda: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      return -1;                                                                                  \
    }                                                                                             \
  } while (0) 

#define CKF(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
  fprintf(stderr, "srs_est_cuda: cuFFT error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); return -1; } } while (0)

/* the largest problem srs_est_init() was asked to support: size your buffers, plans and handles
 * for it once, srs_est_run() is called for every SRS */
static struct {
  int max_rx, max_ports, max_M, max_K, batch;
  float *buffer;
  cufftComplex *d_delay;   // [batch][M]
  cufftComplex *d_full;    // [batch][K*M]
  c16_t *d_y;            // [port][rx][symb][M]
  cufftComplex *d_rbar;  // [symb][M]
  cufftComplex *h_taps;  // host copy of d_delay, for the reductions
  
  cudaStream_t stream;
  bool warned;
} g;

static int check_input(const srs_est_in_t *in)
{
  if (in->nb_rx < 1 || in->nb_rx > g.max_rx || in->M < 1 || in->M > g.max_M || in->K_TC > g.max_K)
    return -1;
  if (in->n_ports < 1 || in->n_ports > g.max_ports || in->n_ports > SRS_EST_MAX_PORTS)
    return -1;
  if (in->n_symb < 1 || in->n_symb > SRS_EST_MAX_SYMBOLS || in->n_cs_max < 1)
    return -1;
  return 0;
} 

#define MAX_PLANS 8
static struct { int M, K; cufftHandle inv, fwd; } plans[MAX_PLANS];
static int n_plans;

static int get_plans(int M, int K, cufftHandle *inv, cufftHandle *fwd)
{
  for (int i = 0; i < n_plans; i++)
    if (plans[i].M == M && plans[i].K == K) { *inv = plans[i].inv; *fwd = plans[i].fwd; return 0; }
  if (n_plans == MAX_PLANS) return -1;
  int n1 = M, n2 = K * M;
  cufftHandle a, b;
  CKF(cufftPlanMany(&a, 1, &n1, NULL, 1, n1, NULL, 1, n1, CUFFT_C2C, g.batch));
  if (cufftPlanMany(&b, 1, &n2, NULL, 1, n2, NULL, 1, n2, CUFFT_C2C, g.batch) != CUFFT_SUCCESS) {
    cufftDestroy(a);
    return -1;
  }
  cufftSetStream(a, g.stream);
  cufftSetStream(b, g.stream);
  plans[n_plans++] = {M, K, a, b};
  *inv = a; *fwd = b;
  return 0;
}

extern "C" int32_t srs_est_init(int max_rx, int max_ports, int max_M, int max_K_TC)
{
  g.max_rx = max_rx;
  g.max_ports = max_ports;
  g.max_M = max_M;
  g.max_K = max_K_TC;  
  g.batch = max_rx * max_ports; 

  // uint buffer_s = 4* g.max_rx * g.max_ports * g.max_K * g.max_M; 
  size_t buffer_s = 4 * (size_t)g.max_rx * g.max_ports * g.max_K * g.max_M; 

  float *d_buffer = NULL;
  cudaError_t err = cudaMalloc((void **)&d_buffer, buffer_s); 

  if (err != cudaSuccess) {
    return 1; 
  } 

  g.buffer = d_buffer; 

  CK(cudaMalloc((void **)&g.d_delay, sizeof(cufftComplex) * g.batch * max_M));
  CK(cudaMalloc((void **)&g.d_full,  sizeof(cufftComplex) * g.batch * max_K_TC * max_M));
  CK(cudaStreamCreate(&g.stream));
  CK(cudaMalloc((void **)&g.d_y, sizeof(c16_t) * max_ports * max_rx * SRS_EST_MAX_SYMBOLS * max_M));
  CK(cudaMalloc((void **)&g.d_rbar, sizeof(cufftComplex) * SRS_EST_MAX_SYMBOLS * max_M));
  g.h_taps = (cufftComplex *)malloc(sizeof(cufftComplex) * g.batch * max_M);
  if (!g.h_taps) return -1;

  cufftHandle inv, fwd; 

  if (get_plans(max_M, max_K_TC, &inv, &fwd) != 0){
    return -1; 
  };

  // TODO: allocate the device buffers, create the FFT plans and solver handles you need
  return 0;
}

extern "C" void srs_est_shutdown(void)
{
  // TODO: free what srs_est_init() allocated
  if (g.stream)
    cudaStreamSynchronize(g.stream);
  for (int i = 0; i < n_plans; i++) {
    cufftDestroy(plans[i].inv);
    cufftDestroy(plans[i].fwd);
  }
  n_plans = 0;
  cudaFree(g.d_delay); g.d_delay = NULL;
  cudaFree(g.d_full);  g.d_full = NULL;
  cudaFree(g.d_y);    g.d_y = NULL;
  cudaFree(g.d_rbar); g.d_rbar = NULL;
  free(g.h_taps);     g.h_taps = NULL;
  
  if (g.stream)
    cudaStreamDestroy(g.stream);
  g.stream = 0; 


} 

struct ports_t {
  int n_groups;
  int rep[SRS_EST_MAX_PORTS];    // per comb group: a port that carries its y
  int grp[SRS_EST_MAX_PORTS];    // per port: its comb group
  int shift[SRS_EST_MAX_PORTS];  // per port: n_cs * M / n_cs_max, in delay bins
};

// z[grp][a][k] = (1/n_symb) sum_l y[rep][a][l][k] * conj(rbar[l][k])
__global__ void ls_kernel(const c16_t *y, const float2 *rbar, float2 *z,
                          ports_t P, int nb_rx, int n_symb, int M)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;                       // grp * nb_rx + a
  if (k >= M) return;
  int grp = row / nb_rx, a = row % nb_rx, p = P.rep[grp];
  float2 acc = {0.f, 0.f};
  for (int l = 0; l < n_symb; l++) {
    c16_t v = y[(((size_t)p * nb_rx + a) * n_symb + l) * M + k];
    float2 r = rbar[l * M + k];
    acc.x += v.r * r.x + v.i * r.y;           // v * conj(r)
    acc.y += v.i * r.x - v.r * r.y;
  }
  z[(size_t)row * M + k] = make_float2(acc.x / n_symb, acc.y / n_symb);
}

// for (port p, antenna a): taps at delays d in [-pre, post] -> zero-padded row of K*M
__global__ void window_kernel(const float2 *taps, float2 *full, ports_t P,
                              int nb_rx, int M, int K, int pre, int post)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. pre+post
  int row = blockIdx.y;                           // p * nb_rx + a
  if (i > pre + post) return;
  int p = row / nb_rx, a = row % nb_rx;
  int d = i - pre;                                // delay in bins
  int src = ((d - P.shift[p]) % M + M) % M;       // where port p's tap d landed
  float2 v = taps[((size_t)P.grp[p] * nb_rx + a) * M + src];
  int N = K * M;
  int dst = d >= 0 ? d : N + d;                   // negative delays wrap to the end
  full[(size_t)row * N + dst] = make_float2(v.x / M, v.y / M);  // 1/M: cuFFT is unnormalised
}

extern "C" int32_t srs_est_run(const srs_est_in_t *in, srs_est_out_t *out)
{
  if (check_input(in) != 0)
    return -1; 

  const int M = in->M, K = in->K_TC, N = K * M, nb_rx = in->nb_rx, np = in->n_ports;

  cufftHandle inv, fwd;
  if (get_plans(in->M, in->K_TC, &inv, &fwd) != 0)
    return -1;

  // comb groups and per-port delay shifts
  ports_t P = {};
  for (int p = 0; p < np; p++) {
    int gi = -1;
    for (int q = 0; q < p; q++)
      if (in->comb[q] == in->comb[p]) { gi = P.grp[q]; break; }
    if (gi < 0) { gi = P.n_groups++; P.rep[gi] = p; }
    P.grp[p] = gi;
    P.shift[p] = in->n_cs[p] * M / in->n_cs_max;
  }
  int pre = in->win_pre, post = in->win_post;
  int per_grp = np / P.n_groups; if (per_grp < 1) per_grp = 1;
  int spacing = M / in->n_cs_max * (in->n_cs_max / per_grp);
  if (pre + post + 1 > spacing) return -1;   // windows would overlap: let the gNB fall back

  // 1. inputs
  CK(cudaMemcpyAsync(g.d_y, in->y, sizeof(c16_t) * np * nb_rx * in->n_symb * M, cudaMemcpyHostToDevice, g.stream));
  CK(cudaMemcpyAsync(g.d_rbar, in->rbar, sizeof(cf_t) * in->n_symb * M, cudaMemcpyHostToDevice, g.stream));

  // 2. LS + average
  CK(cudaMemsetAsync(g.d_delay, 0, sizeof(cufftComplex) * g.batch * M, g.stream));
  dim3 b(256), gr1((M + 255) / 256, P.n_groups * nb_rx);
  ls_kernel<<<gr1, b, 0, g.stream>>>(g.d_y, g.d_rbar, g.d_delay, P, nb_rx, in->n_symb, M);

  // 3. delay domain
  CKF(cufftExecC2C(inv, g.d_delay, g.d_delay, CUFFT_INVERSE));

  // 4. taps to host for the reductions (overlaps with 5-6 on the GPU)
  CK(cudaMemcpyAsync(g.h_taps, g.d_delay, sizeof(cufftComplex) * P.n_groups * nb_rx * M, cudaMemcpyDeviceToHost, g.stream));

  // 5. window -> padded rows
  CK(cudaMemsetAsync(g.d_full, 0, sizeof(cufftComplex) * g.batch * N, g.stream));
  dim3 gr2((pre + post + 1 + 255) / 256, np * nb_rx);
  window_kernel<<<gr2, b, 0, g.stream>>>(g.d_delay, g.d_full, P, nb_rx, M, K, pre, post);
  CK(cudaGetLastError());

  // 6. h_full, and h_comb = every K-th
  CKF(cufftExecC2C(fwd, g.d_full, g.d_full, CUFFT_FORWARD));

  // 7. outputs
  size_t rows = (size_t)np * nb_rx;
  CK(cudaMemcpyAsync(out->h_full, g.d_full, sizeof(cf_t) * rows * N, cudaMemcpyDeviceToHost, g.stream));
  CK(cudaMemcpy2DAsync(out->h_comb, sizeof(cf_t), g.d_full, K * sizeof(cufftComplex),
                       sizeof(cf_t), rows * M, cudaMemcpyDeviceToHost, g.stream));
  CK(cudaStreamSynchronize(g.stream));

  // noise: taps outside every port's window. Unscaled IFFT -> per-bin variance M*sigma^2/n_symb
  double noise = 0; int n_noise = 0;
  double pdp_best = -1; int d_best = 0;
  for (int gi = 0; gi < P.n_groups; gi++)
    for (int n = 0; n < M; n++) {
      int in_win = 0, d_here = 0;
      for (int p = 0; p < np; p++) {
        if (P.grp[p] != gi) continue;
        int d = ((n + P.shift[p] + pre) % M) - pre;   // delay of bin n relative to port p
        if (d >= -pre && d <= post) { in_win = 1; d_here = d; }
      }
      double pw = 0;
      for (int a = 0; a < nb_rx; a++) {
        cufftComplex t = g.h_taps[((size_t)gi * nb_rx + a) * M + n];
        pw += (double)t.x * t.x + (double)t.y * t.y;
      }
      if (in_win) { if (pw > pdp_best) { pdp_best = pw; d_best = d_here; } }
      else { noise += pw; n_noise += nb_rx; }
    }
  out->noise_var = n_noise ? (float)(noise / n_noise * in->n_symb / M) : 0.f;
  out->timing_offset_s = d_best / ((double)N * in->scs_hz);

  double sp = 0;
  for (size_t i = 0; i < rows * M; i++)
    sp += (double)out->h_comb[i].r * out->h_comb[i].r + (double)out->h_comb[i].i * out->h_comb[i].i;
  out->signal_power = (float)(sp / (rows * M));
  return 0;
}
  // TODO: estimate the channel of every port at every antenna from in->y and in->rbar, and fill
  // out->h_comb, out->h_full, out->noise_var, out->signal_power and out->timing_offset_s.
  // Return 0 once they are filled: the gNB then uses them instead of its fixed-point estimate.
  // (void)out;
  // if (!g.warned) {
  //   fprintf(stderr, "srs_est_cuda: srs_est_run() is not implemented yet\n");
  //   g.warned = true;
  // }
  // return -1; 


