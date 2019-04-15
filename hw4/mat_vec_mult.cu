#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void VVMult_CPU(double* sum_ptr, const double* a, const double* b, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i]*b[i];
  *sum_ptr = sum;
}

void MVMult_CPU(double *C, double *A, double *B, long N) {
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N, j++) {
      double A_ij = A[i*N + j];
      double B_j = B[j];
      double C_i = C[i];
      C_i = C_i + A_ij * B_j;
      C[i] = C_i;
    }
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

__global__ void reduction(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void product(double* sum, const double* A, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = A[idx]*b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

int main() {
  long N = (1UL<<15);

  double *x, *y_ref, *y, *A;
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&A, N*N*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = 1.0/(i+1);
    y[i] = 0;
    y_ref[i] = 0;
  }
  for (long i = 0; i < N*N; i++) {
    A[i] = drand48();
  }

  double tt = omp_get_wtime();
  MVMult_CPU(y_ref, A, x, N);
  printf("CPU Bandwidth = %f GB/s\n", 3*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *z_d;
  double *A_d;
  long N_work = 1;
  for (long i = (N*N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) {
    N_work += i;
  }
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&A_d, N*N*sizeof(double));
  cudaMalloc(&z_d, N_work*sizeof(double));
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  tt = omp_get_wtime();
  for (long i = 0; i < N; i++) {
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    product<<<Nb,BLOCK_SIZE>>>(z_d, A_d+i*N, x_d, N);
    while (Nb > 1) {
      long N = Nb;
      Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction<<<Nb,BLOCK_SIZE>>>(z_d + N, z_d, N);
      z_d += N;
    }
    cudaMemcpyAsync(y[i], z_d, 1*sizeof(double, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
  }

  printf("GPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  double error = 0;
  for (int i=0; i<N; i++){
	   error += fabs(y[i]-y_ref[i]);
   }
  printf("Error = %f\n", error);

  cudaFree(x_d);
  cudaFree(z_d);
  cudaFree(A_d);
  cudaFreeHost(A);
  cudaFreeHost(x);
  cudaFreeHost(y);

  return 0;
}
