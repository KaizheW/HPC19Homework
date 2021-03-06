/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug.
* AUTHOR: Blaise Barney
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
int nthreads, i, tid;
long total = 0;

/*** Spawn parallel region ***/
#pragma omp parallel private(i, tid)
// Comment:
// For each thread, i and tid should be private.
// Declare that i and tid are private in #pragma omp.
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  // total = 0;
  #pragma omp for schedule(dynamic,10) reduction(+:total)
  // Comment:
  // To parallel calculate the summation, the reduction clause is needed.
  // Add the clause, reduction(+:total).
  for (i=0; i<1000000; i++)
     total = total + i;

  printf ("Thread %d is done! Total= %ld\n",tid,total);

  } /*** End of parallel region ***/
}
