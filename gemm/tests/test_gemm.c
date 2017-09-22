#include <stdio.h>
#include <stdlib.h>

#include "unity.h"
#include "gemm.h"
#include "timer.h"
#include "data.h"


void test_gemm_nn(void)
{
  unsigned long m = 800;
  unsigned long n = 600;
  unsigned long k = 200;
  float alpha = 1;
  float *A;
  unsigned long lda = k;
  float *B;
  unsigned long ldb = n;
  float beta = 0;
  float *C;
  unsigned long ldc = n;

  A = (float*)malloc(m*k*sizeof(A));
  B = (float*)malloc(k*n*sizeof(B));
  C = (float*)malloc(m*n*sizeof(C));

  random_initializer (A, m*k);
  random_initializer (B, k*n);

  double tic = timer();
  gemm_cpu (0, 0, (int)m, (int)n, (int)k, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
  printf ("gemm elapsed: %f\n", timer() - tic);

  free(A);
  free(B);
  free(C);
}


int main(void)
{
  UNITY_BEGIN();
  RUN_TEST (test_gemm_nn);
  return UNITY_END();
}

