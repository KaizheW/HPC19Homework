#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  int nthreads;
  long psize;
  # pragma omp parallel
  {
    int tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    if (tid == 0) printf("Number of threads = %d\n", nthreads);

    psize = n/nthreads;
    long istart = psize*tid + 1;
    long iend = std::min(psize*(tid+1)+1, n);
    // #pragma omp for
    for (long i = istart; i < iend; i++) {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
  }
  for (int j = 1; j < nthreads; j++) {
    long jstart = psize*(long)j + 1;
    long jend = std::min(psize*((long)j+1)+1,n);
    for (long i = jstart; i < jend; i++) {
      prefix_sum[i] = prefix_sum[i] + prefix_sum[jstart-1];
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
