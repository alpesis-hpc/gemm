#include <stdio.h>
#include <stdlib.h>

#include "unity.h"
#include "timer.h"
#include "data.h"
#include "gemm_pthread.h"


void test_gemm_pthread(void)
{
  unsigned long m = 800;
  unsigned long n = 600;
  unsigned long k = 200;
  float alpha = 1;
  float beta = 1;
  unsigned long lda = k;
  unsigned long ldb = n;
  unsigned long ldc = n;

  float * A = (float*)malloc(m*k*sizeof(A));
  float * B = (float*)malloc(k*n*sizeof(B));
  float * C = (float*)malloc(m*n*sizeof(C));

  random_initializer (A, m*k);
  random_initializer (B, k*n);

  double tic;
  tic = timer();
  thread_gemm_cpu (0, 0, (int)m, (int)n, (int)k, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
  printf ("gemm_nn elapsed: %f\n", timer() - tic);

  tic = timer();
  thread_gemm_cpu (0, 1, (int)m, (int)n, (int)k, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
  printf ("gemm_nt elapsed: %f\n", timer() - tic);

  tic = timer();
  thread_gemm_cpu (1, 0, (int)m, (int)n, (int)k, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
  printf ("gemm_tn elapsed: %f\n", timer() - tic);

  tic = timer();
  thread_gemm_cpu (1, 1, (int)m, (int)n, (int)k, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
  printf ("gemm_tt elapsed: %f\n", timer() - tic);

  free(A);
  free(B);
  free(C);
}


int main(void)
{
  UNITY_BEGIN();
  RUN_TEST (test_gemm_pthread);
  return UNITY_END();
}

