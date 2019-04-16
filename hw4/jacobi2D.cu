#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <math.h>

#define FWIDTH 3
#define BLOCK_DIM 32

__constant__ float filter[FWIDTH][FWIDTH] = {
  0,    0.25, 0,
  0.25, 0,    0.25,
  0,    0.25, 0};

__global__ void jacobi_GPU(double *uk, double *u, double *f, long N) {
  constexpr double h = 1.0/(N+1.0);
  __shared__ float smem[BLOCK_DIM+FWIDTH][BLOCK_DIM+FWIDTH];
  long offset_x = blockIdx.x * (BLOCK_DIM - FWIDTH);
  long offset_y = blockIdx.y * (BLOCK_DIM - FWIDTH);

  smem[threadIdx.x][threadIdx.y] = 0;
  if (offset_x + threadIdx.x < N+2 && offset_y + threadIdx.y < N+2)
    smem[threadIdx.x][threadIdx.y] = u[(offset_x + threadIdx.x)*(N+2) + (offset_y + threadIdx.y)];
  __syncthreads();

  double sum = 0;
  for (int j0 = 0; j0 < FWIDTH; j0++) {
    for (int j1 = 0; j1 < FWIDTH; j1++) {
      sum += smem[threadIdx.x+j0][threadIdx.y+j1] * filter[j0][j1];
    }
  }
  sum = sum + 0.25*h*h*f[(offset_x + threadIdx.x)*(N+2) + (offset_y + threadIdx.y)];

  if (threadIdx.x+FWIDTH < BLOCK_DIM && threadIdx.y+FWIDTH < BLOCK_DIM)
    if (offset_x+threadIdx.x+FWIDTH <= Xsize && offset_y+threadIdx.y+FWIDTH <= Ysize)
      uk[(offset_x+threadIdx.x+1)*(N+2) + (offset_y+threadIdx.y+1)] = (double)fabs(sum);

}

int main() {
  long N = 128;
  long max_iter = 100;
  printf("Matrix size: %d\n", N);

  double *u, *uk, *f;
  cudaMallocHost((void**)&u , (N+2)*(N+2)*sizeof(double));
  cudaMallocHost((void**)&uk, (N+2)*(N+2)*sizeof(double));
  cudaMallocHost((void**)&f , (N+2)*(N+2)*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (N+2)*(N+2); i++) {
    u[i] = 0;
    f[i] = 0;
    uk[i] = 0;
  }

  double *u_d, *uk_d, *f_d;
  cudaMalloc(&u_d , (N+2)*(N+2)*sizeof(double));
  cudaMalloc(&uk_d, (N+2)*(N+2)*sizeof(double));
  cudaMalloc(&f_d , (N+2)*(N+2)*sizeof(double));
  cudaMemcpyAsync(u_d, u, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(uk_d, uk, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(f_d, f, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  tt = omp_get_wtime();
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim((N+2)/(BLOCK_DIM-FWIDTH)+1, (N+2)/(BLOCK_DIM-FWIDTH)+1);
  for (int iter = 0; iter < max_iter; iter++) {
    jacobi_GPU<<<gridDim, blockDim>>>(uk_d, u_d, f_d, N);
    cudaMemcpyAsync(u_d, uk_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
  }
  cudaMemcpyAsync(&u, u_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
}
