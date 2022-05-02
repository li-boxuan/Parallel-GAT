#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <limits>

#include "CycleTimer.h"
#include "sparse.h"

#define TILE_WIDTH 32
#define TILE_SIZE (TILE_WIDTH * TILE_WIDTH)
#define RM(r, c, width) ((r) * (width) + (c))

__global__ void mm_kernel(int m, int p, int n, float *device_A, float *device_B, float *device_C) {
  // (m, p) * (p, n)
  // blockDim.y == blockDim.x == TILE_WIDTH
  int row_start = blockDim.y * blockIdx.y;
  int col_start = blockDim.y * blockIdx.x;
  int row_offset = threadIdx.y;
  int col_offset = threadIdx.x;
  float res = 0.f;
  __shared__ float A[TILE_WIDTH * TILE_WIDTH];
  __shared__ float B[TILE_WIDTH * TILE_WIDTH];
  for (int offset = 0; offset < p; offset += TILE_WIDTH) {
    int A_row_idx = row_start + row_offset;
    int A_col_idx = offset + col_offset;
    int B_row_idx = offset + row_offset;
    int B_col_idx = col_start + col_offset;
    if (A_row_idx < m && A_col_idx < p) {
      A[row_offset * TILE_WIDTH + col_offset] = device_A[p * A_row_idx + A_col_idx];
    } else {
      A[row_offset * TILE_WIDTH + col_offset] = 0.f;
    }
    if (B_row_idx < p && B_col_idx < n) {
      B[row_offset * TILE_WIDTH + col_offset] = device_B[n * B_row_idx + B_col_idx];
    } else {
      B[row_offset * TILE_WIDTH + col_offset] = 0.f;
    }
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++) {
      res += A[row_offset * TILE_WIDTH + k] * B[k * TILE_WIDTH + col_offset];
    }
    __syncthreads();
  }
  int res_row_idx = row_start + row_offset;
  int res_col_idx = col_start + col_offset;
  if (res_row_idx < m && res_col_idx < n) {
    device_C[res_row_idx * n + res_col_idx] = res;
  }
}

__global__ void heat_kernel(int num_nodes, int num_heads, int out_dim,
                            float *device_msgs, float *device_A, float *device_heats) {
  // device_A: (num_heads, 2, out_dim)
  // device_msgs: (num_nodes, num_heads, out_dim)
  // device_heats: (num_heads, 2, num_nodes)
  // grid: (num_heads, num_nodes / tile_width)
  // block: (2, tile_width)
  extern __shared__ float A[];
  int head_idx = blockIdx.x;
  int block_size = blockDim.x * blockDim.y;
  int idx_within_block = threadIdx.x * blockDim.y + threadIdx.y;
  // load device_A[head_idx, :, :]
  int A_start_idx = head_idx * 2 * out_dim;
  int node_idx = blockDim.y * blockIdx.y + threadIdx.y;
  int num_per_thread = (2 * out_dim + block_size - 1) / block_size;
  for (int k = 0; k < num_per_thread; k++) {
    int local_idx = k + idx_within_block * num_per_thread;
    if (local_idx < 2 * out_dim) {
      A[local_idx] = device_A[A_start_idx + local_idx];
    }
  }
  __syncthreads();
  // device_msgs[node_idx, head_idx, :] * A[threadIdx.x, :]
  float res = 0.f;
  if (node_idx < num_nodes) {
    for (int k = 0; k < out_dim; k++) {
      res += device_msgs[node_idx * num_heads * out_dim + head_idx * out_dim + k] * A[threadIdx.x * out_dim + k];
    }
    // device_heats[head_idx, threadIdx.x, node_idx]
    device_heats[head_idx * 2 * num_nodes + threadIdx.x * num_nodes + node_idx] = res;
  }
}

__global__ void attn_kernel(int num_nodes, int num_elems, int *col_idx, int *delim,
                            float *heats, float *attn, float min_f) {
  // heats: (num_heads, 2, num_nodes)
  // attn: (num_heads, num_elems)
  // col_idx: (num_elems,)
  // delim: (num_nodes,)
  // grid: (num_heads, num_nodes / tile_size)
  // block: (tile_size)
  int head_idx = blockIdx.x;
  int node_idx = blockDim.y * blockIdx.y + threadIdx.x;
  if (node_idx >= num_nodes) {
    return;
  }
  float max_affinity = min_f;
  // heats[head_idx, 0, node_idx]
  float curr_node_heat = heats[head_idx * 2 * num_nodes + node_idx];

  int col_start = delim[node_idx];
  int col_end = delim[node_idx + 1];
  for (int k = col_start; k < col_end; k++) {
    int neighbor_idx = col_idx[k];
    // heats[head_idx, 1, neighbor_idx]
    float neighbor_node_heat = heats[head_idx * 2 * num_nodes + num_nodes + neighbor_idx];
    float heat_sum = curr_node_heat + neighbor_node_heat;
    float curr_affinity = (heat_sum > 0.f) ? heat_sum : (0.2f * heat_sum);
    if (curr_affinity > max_affinity) max_affinity = curr_affinity;
    // attn[head_idx, k]
    attn[head_idx * num_elems + k] = curr_affinity;
  }
  float affinity_sum = 0.f;
  for (int k = col_start; k < col_end; k++) {
    float curr_affinity = exp(attn[head_idx * num_elems + k] - max_affinity);
    affinity_sum += curr_affinity;
    attn[head_idx * num_elems + k] = curr_affinity;
  }
  for (int k = col_start; k < col_end; k++) {
    attn[head_idx * num_elems + k] /= affinity_sum;
  }

}

__global__ void aggregate_kernel(int num_nodes, int num_heads, int out_dim, int num_elems,
                                 float *device_msgs, float *device_attn,
                                 int *device_col_idx, int *device_delim, float *device_output_feats) {
  // grid: (num_heads, (num_nodes * out_dim) / tile_size)
  // block: (tile_size)
  // device_out_feats: (num_nodes, num_head, out_dim)
  // device_msgs: (num_nodes, num_heads, out_dim)
  // device_attn: (num_heads, num_elems)
  // device_col_idx: (num_elems,)
  // device_delim: (num_nodes, )

  int head_idx = blockIdx.x;
  int idx_within_head = blockIdx.y * TILE_SIZE + threadIdx.x;
  int node_idx = idx_within_head / out_dim;
  int local_feat_idx = idx_within_head % out_dim;
  int feat_idx = head_idx * out_dim + local_feat_idx;
  int neighbor_start = device_delim[node_idx];
  int neighbor_end = device_delim[node_idx + 1];
  if (idx_within_head < num_nodes * out_dim) {
    int global_idx = node_idx * num_heads * out_dim + feat_idx;
    device_output_feats[global_idx] = 0.f;
    for (int k = neighbor_start; k < neighbor_end; k++) {
      int neighbor_idx = device_col_idx[k];
      float w = device_attn[num_elems * head_idx + k];
      device_output_feats[global_idx] +=
          w * device_msgs[neighbor_idx * (num_heads * out_dim) + head_idx * out_dim + local_feat_idx];
    }
  }
}

void gatForwardCUDA(float *W, float *A, float *input_feats, sparse_matrix *adj, int in_dim,
                    int out_dim, int num_heads, int num_nodes, float *output_feats, float min_f) {
  float *device_W;
  float *device_input_feats;
  float *device_msgs;
  double startTime = CycleTimer::currentSeconds();

  cudaMalloc((void **) &device_W, in_dim * num_heads * out_dim * sizeof(float));
  cudaMalloc((void **) &device_input_feats, num_nodes * in_dim * sizeof(float));
  cudaMalloc((void **) &device_msgs, num_nodes * num_heads * out_dim * sizeof(float));


  cudaMemcpy(device_W, W, in_dim * num_heads * out_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_input_feats, input_feats, num_nodes * in_dim * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock = dim3(TILE_WIDTH, TILE_WIDTH);
  dim3 blocks = dim3((num_nodes + TILE_WIDTH - 1) / TILE_WIDTH, out_dim * num_heads + TILE_WIDTH - 1 / TILE_WIDTH);
  mm_kernel<<<blocks, threadsPerBlock>>>(num_nodes, in_dim, num_heads * out_dim, device_input_feats, device_W,
                                         device_msgs);
  cudaDeviceSynchronize();
//  double endTime = CycleTimer::currentSeconds();
//  double overallDuration = endTime - startTime;
////  printf("Kernel invocation: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));
//  printf("Overall: %.3f ms\n", 1000.f * overallDuration);
//  float *test_msgs;
//  test_msgs = new float[num_nodes * num_heads * out_dim];
//  cudaMemcpy(test_msgs, device_msgs, num_nodes * num_heads * out_dim * sizeof(float), cudaMemcpyDeviceToHost);
//  printf("msgs %.3f %.3f %.3f %.3f\n", test_msgs[0], test_msgs[1], test_msgs[2], test_msgs[3]);
//  delete(test_msgs);
  cudaFree(device_W);
  cudaFree(device_input_feats);

  float *device_A;
  float *device_heats;
  cudaMalloc((void **) &device_A, num_heads * 2 * out_dim * sizeof(float));
  cudaMalloc((void **) &device_heats, num_heads * 2 * num_nodes * sizeof(float));

  cudaMemcpy(device_A, A, num_heads * 2 * out_dim * sizeof(float), cudaMemcpyHostToDevice);

  threadsPerBlock = dim3(2, TILE_WIDTH);
  blocks = dim3(num_heads, (num_nodes + TILE_WIDTH - 1) / TILE_WIDTH);

  heat_kernel<<<blocks, threadsPerBlock, 2 * out_dim * sizeof(float)>>>(num_nodes, num_heads, out_dim, device_msgs,
                                                                        device_A, device_heats);
  cudaDeviceSynchronize();

//  float *test_heats;
//  test_heats = new float[num_heads * 2 * num_nodes];
//  cudaMemcpy(test_heats, device_heats, num_heads * 2 * num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
//  printf("heats %.3f %.3f %.3f %.3f\n", test_heats[0], test_heats[1], test_heats[2], test_heats[3]);
//  delete(test_heats);

  cudaFree(device_A);

  float *device_attn;
  int *device_col_idx;
  int *device_delim;
  cudaMalloc((void **) &device_attn, num_heads * adj->num_elements * sizeof(float));
  cudaMalloc((void **) &device_col_idx, adj->num_elements * sizeof(int));
  cudaMalloc((void **) &device_delim, (adj->num_rows + 1) * sizeof(int));

  cudaMemcpy(device_col_idx, adj->col_idx, adj->num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_delim, adj->delim, (adj->num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);


  threadsPerBlock = dim3(TILE_SIZE);
  blocks = dim3(num_heads, (num_nodes + TILE_SIZE - 1) / TILE_SIZE);

  attn_kernel<<<blocks, threadsPerBlock>>>(num_nodes, adj->num_elements,
                                           device_col_idx, device_delim, device_heats, device_attn, min_f);
  cudaDeviceSynchronize();


//  float *test_attn;
//  test_attn = new float[num_heads * adj->num_elements];
//  cudaMemcpy(test_attn, device_attn, num_heads * adj->num_elements * sizeof(float), cudaMemcpyDeviceToHost);
//  printf("attn %.3f %.3f %.3f %.3f\n", test_attn[0], test_attn[1], test_attn[2], test_attn[3]);
//  delete(test_attn);
  cudaFree(device_heats);

  float *device_output_feats;
  cudaMalloc((void **) &device_output_feats, num_nodes * num_heads * out_dim * sizeof(float));

  threadsPerBlock = dim3(TILE_SIZE);
  blocks = dim3(num_heads, (num_nodes * out_dim + TILE_SIZE - 1) / TILE_SIZE);
  aggregate_kernel<<<blocks, threadsPerBlock>>>(num_nodes, num_heads, out_dim, adj->num_elements,
                                                device_msgs, device_attn, device_col_idx, device_delim,
                                                device_output_feats);
  cudaDeviceSynchronize();

//  float *test_output_feats;
//  test_output_feats = new float[num_nodes * num_heads * out_dim];
//  cudaMemcpy(test_output_feats, device_output_feats, num_nodes * num_heads * out_dim * sizeof(float), cudaMemcpyDeviceToHost);
//  printf("output_feats %.3f %.3f %.3f %.3f\n", test_output_feats[0], test_output_feats[1], test_output_feats[2], test_output_feats[3]);
//  delete(test_output_feats);

  cudaFree(device_msgs);
  cudaFree(device_col_idx);
  cudaFree(device_delim);

  cudaMemcpy(output_feats, device_output_feats, num_heads * out_dim * num_nodes * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(device_output_feats);

  double endTime = CycleTimer::currentSeconds();

  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
  }

  double overallDuration = endTime - startTime;
//  printf("Kernel invocation: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));
  printf("Overall: %.3f ms\n", 1000.f * overallDuration);
}


void printCudaInfo() {
  // For fun, just print out some stats on the machine

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}
