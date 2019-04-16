#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <math.h>

#define FWIDTH 3
#define BLOCK_DIM 32
#define BLOCK_SIZE 1024

__constant__ float filter[FWIDTH][FWIDTH] = {
  0,    0.25, 0,
  0.25, 0,    0.25,
  0,    0.25, 0};

__constant__ float laplace[FWIDTH][FWIDTH] = {
  0 , -1, 0 ,
  -1, 4 , -1,
  0 , -1, 0 };

__global__ void jacobi_GPU(double *uk, double *u, double *f, long N) {
  double h = 1.0/(N+1.0);
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
    if (offset_x+threadIdx.x+FWIDTH <= (N+2) && offset_y+threadIdx.y+FWIDTH <= (N+2))
      uk[(offset_x+threadIdx.x+1)*(N+2) + (offset_y+threadIdx.y+1)] = (double)fabs(sum);

}

__global__ void res_GPU(double *res, double *u, double *f, long N) {
  double h = 1.0/(N+1.0);
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
      sum += smem[threadIdx.x+j0][threadIdx.y+j1] * laplace[j0][j1];
    }
  }
  sum = sum /(h*h)-f[(offset_x + threadIdx.x)*(N+2) + (offset_y + threadIdx.y)];


  if (threadIdx.x+FWIDTH < BLOCK_DIM && threadIdx.y+FWIDTH < BLOCK_DIM)
    if (offset_x+threadIdx.x+FWIDTH <= (N+2) && offset_y+threadIdx.y+FWIDTH <= (N+2))
      res[(offset_x+threadIdx.x+1)*(N+2) + (offset_y+threadIdx.y+1)] = (double)fabs(sum);

}

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

int main() {
  long N = (1UL<<10);
  long max_iter = 10000;
  printf("Matrix size: %d\n", N);
  printf("Maximum Iteration Number: %d\n", max_iter);

  double *u, *uk, *f;
  cudaMallocHost((void**)&u, (N+2)*(N+2)*sizeof(double));
  cudaMallocHost((void**)&uk, (N+2)*(N+2)*sizeof(double));
  cudaMallocHost((void**)&f, (N+2)*(N+2)*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < (N+2)*(N+2); i++) {
    u[i] = 0;
    f[i] = 1;
    uk[i] = 0;
  }

  double *u_d, *uk_d, *f_d, *res_d;
  cudaMalloc(&u_d , (N+2)*(N+2)*sizeof(double));
  cudaMalloc(&uk_d, (N+2)*(N+2)*sizeof(double));
  cudaMalloc(&f_d , (N+2)*(N+2)*sizeof(double));
  cudaMalloc(&res_d, (N+2)*(N+2)*sizeof(double));
  cudaMemcpyAsync(u_d, u, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(uk_d, uk, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(f_d, f, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  double tt = omp_get_wtime();
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim((N+2)/(BLOCK_DIM-FWIDTH)+1, (N+2)/(BLOCK_DIM-FWIDTH)+1);
  for (int iter = 0; iter < max_iter; iter++) {
    jacobi_GPU<<<gridDim, blockDim>>>(uk_d, u_d, f_d, N);
    cudaMemcpyAsync(u_d, uk_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
  }
  res_GPU<<<gridDim, blockDim>>>(res_d, u_d, f_d, N);

  double res;
  double *y_d;
  long N_work = 1;
  for (long i = ((N+2)*(N+2)+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&y_d, N_work*sizeof(double));
  long Nb = ((N+2)*(N+2)+BLOCK_SIZE-1)/(BLOCK_SIZE);
  reduction<<<Nb,BLOCK_SIZE>>>(y_d, res_d, (N+2)*(N+2));
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction<<<Nb,BLOCK_SIZE>>>(y_d + N, y_d, N);
    y_d += N;
  }


  cudaMemcpyAsync(&res, y_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
//  printf("GPU Bandwidth = %f GB/s\n", max_iter*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("time: %f\n", omp_get_wtime()-tt);

  printf("res =  %f\n", res);
  cudaFree(u_d);
  cudaFree(uk_d);
  cudaFree(f_d);
  cudaFree(res_d);
  cudaFree(y_d);
  cudaFreeHost(u);
  cudaFreeHost(uk);
  cudaFreeHost(f);

  return 0;  
}
