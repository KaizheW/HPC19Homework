#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

void Jacobi(long n, long max_iter, double *u, double *f) {
  double h = 1.0 / (n+1.0);
  int i, j;
  long p;
  double res;
  double lu; // minus laplace u
  double* uk = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double));
  // printf(" Iteration        Res\n");
  for (int iter = 0; iter < max_iter; iter++) {
    res = 0.0;
    // #pragma omp parallel private(p,i,j) shared(u,uk)
    // {
    #pragma omp parallel for
    for (p = 0; p < n*n; p++) {
      i = p % n + 1;
      j = p / n + 1;
      uk[(n+2)*j + i] = 0.25 * (h*h*f[(n+2)*j + i] + u[(n+2)*j + i+1] \
        + u[(n+2)*j + i-1] + u[(n+2)*(j-1) + i] + u[(n+2)*(j+1) + i]);
    }
    #pragma omp barrier

    // #pragma omp parallel private(p,i,j) shared(u,uk)
    #pragma omp parallel for
    for (long p = 0; p < n*n; p++) {
      i = p % n + 1;
      j = p / n + 1;
      u[(n+2)*j + i] = uk[(n+2)*j + i];
    }
    #pragma omp barrier

    #pragma omp parallel for reduction(+:res)
    for (long p = 0; p < n*n; p++) {
      i = p % n + 1;
      j = p / n + 1;
      lu = (4.0*u[(n+2)*j + i] - u[(n+2)*j + i+1] - u[(n+2)*j + i-1] - \
        u[(n+2)*(j-1) + i] - u[(n+2)*(j+1) + i])/(h*h);
      res = res + (lu - f[(n+2)*j + i]) * (lu - f[(n+2)*j + i]);
    }
    #pragma omp barrier
    // }
    res = sqrt(res);
    // printf("%5d      %10f\n", iter, res);

  }
  printf("After %d iterations, the residual is %f\n",max_iter, res);
  aligned_free(uk);
}

int main(int argc, char** argv) {
  long n = 100;
  long max_iter = 1000;
  printf("Matrix size: %d\n", n);

  double* u = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double));
  double* f = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double));
  for (int i = 0; i < (n+2)*(n+2); i++) {
    u[i] = 0.0;
    f[i] = 1.0;
  }

  Timer t;
  t.tic();
  Jacobi(n,max_iter,u,f);
  double time = t.toc();
  printf("Total time: %f\n", time);
  // printf("%f\n", u[1050]);

  aligned_free(u);
  aligned_free(f);
  return 0;
}
