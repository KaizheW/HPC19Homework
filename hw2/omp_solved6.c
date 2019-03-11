/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];
float sum;

float dotprod ()
{
  int i,tid;
  // float sum;
  // Comments:
  // sum should be a shared variable, so it can't be defined here.

  tid = omp_get_thread_num();
  #pragma omp for reduction(+:sum)
    for (i=0; i < VECLEN; i++)
      {
        sum = sum + (a[i]*b[i]);
        printf("  tid= %d i=%d\n",tid,i);
      }
  return 0;
}


int main (int argc, char *argv[]) {
int i;
// float sum = 0.0;
// Comments:
// Set sum as a global variable, used in both main and dotprod.

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

#pragma omp parallel shared(sum)
  dotprod();

printf("Sum = %f\n",sum);

}
