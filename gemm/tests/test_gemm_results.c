#include <stdio.h>
#include <stdlib.h>

#include "unity.h"
#include "timer.h"
#include "data.h"

#include "gemm.h"
#include "gemm_pthread.h"


void result_eval (float * C, float * C_pthread, int count)
{
  int i;
  for (i = 0; i < count; ++i)
  {
    if (C[i] - C_pthread[i] != 0)
    {
      printf ("Error: C[%d] %f != C_pthread[%d] %f\n", i, C[i], i, C_pthread[i]);
    }
  }
}


void gemm_compute (int ta, int tb,
                   int m, int n, int k,
                   float alpha,
                   float * A, int lda,
                   float * B, int ldb,
                   float beta,
                   int ldc)
{
  double tic;
  float * C = (float*)malloc(m*n*sizeof(float));
  float * C_pthread = (float*)malloc(m*n*sizeof(float));

  tic = timer();
  gemm_cpu (ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  printf ("- gemm elapsed: %f\n", timer() - tic);

  tic = timer();
  thread_gemm_cpu (ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C_pthread, ldc);
  printf ("- thread_gemm elapsed: %f\n", timer() - tic);

  result_eval (C, C_pthread, m*n);

  free (C);
  free (C_pthread);
}


void test_gemm(void)
{
  unsigned long m = 800;
  unsigned long n = 600;
  unsigned long k = 200;
  float alpha = 1;
  float beta = 1;
  unsigned long lda = k;
  unsigned long ldb = n;
  unsigned long ldc = n;

  float * A = (float*)malloc(m*k*sizeof(float));
  float * B = (float*)malloc(k*n*sizeof(float));
  random_initializer (A, m*k);
  random_initializer (B, k*n);

  printf ("gemm_nn:\n");
  gemm_compute (0, 0, m, n, k, alpha, A, lda, B, ldb, beta, ldc);
  printf ("gemm_tn:\n");
  gemm_compute (1, 0, m, n, k, alpha, A, lda, B, ldb, beta, ldc);
  printf ("gemm_nt:\n");
  gemm_compute (0, 1, m, n, k, alpha, A, lda, B, ldb, beta, ldc);
  printf ("gemm_tt:\n");
  gemm_compute (1, 1, m, n, k, alpha, A, lda, B, ldb, beta, ldc);

  free(A);
  free(B);
}


int main(void)
{
  UNITY_BEGIN();
  RUN_TEST (test_gemm);
  return UNITY_END();
}
