#include "gemm_pthread.h"


void thread_gemm (int ta, int tb, 
                  int M, int N, int K, 
                  float alpha, 
                  float * A, int lda, 
                  float * B, int ldb,
                  float beta,
                  float * C, int ldc)
{
  thread_gemm_cpu (ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}


void thread_gemm_cpu (int ta, int tb, 
                      int M, int N, int K, 
                      float alpha, 
                      float * A, int lda, 
                      float * B, int ldb,
                      float beta,
                      float * C, int ldc)
{
  if      (!ta && !tb)      thread_gemm_nn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
  else if (ta && !tb)       thread_gemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
  else if (!ta && tb)       thread_gemm_nt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
  else                      thread_gemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);

  thread_gemm_beta (M, N, beta, C);
}


void thread_gemm_nn (int M, int N, int K, 
                     float alpha, 
                     float * A, int lda, 
                     float * B, int ldb,
                     float * C, int ldc)
{
  int i, j, k;
  for(i = 0; i < M; ++i)
  {
    for(k = 0; k < K; ++k)
    {
      register float a_part = alpha * A[i*lda+k];
      for(j = 0; j < N; ++j)
      {
        C[i*ldc+j] += a_part * B[k*ldb+j];
      }
    }
  }
}


void thread_gemm_nt (int M, int N, int K, 
                     float alpha, 
                     float * A, int lda, 
                     float * B, int ldb,
                     float * C, int ldc)
{
  int i,j,k;
  for(i = 0; i < M; ++i)
  {
    for(j = 0; j < N; ++j)
    {
      register float sum = 0;
      for(k = 0; k < K; ++k)
      {
        sum += alpha * A[i*lda+k] * B[j*ldb + k];
      }
      C[i*ldc+j] += sum;
    }
  }
}


void thread_gemm_tn (int M, int N, int K, 
                     float alpha, 
                     float * A, int lda, 
                     float * B, int ldb,
                     float * C, int ldc)
{
  int i,j,k;
  for(i = 0; i < M; ++i)
  {
    for(k = 0; k < K; ++k)
    {
      register float a_part = alpha * A[k*lda+i];
      for(j = 0; j < N; ++j)
      {
        C[i*ldc+j] += a_part * B[k*ldb+j];
      }
    }
  }
}


void thread_gemm_tt (int M, int N, int K, 
                     float alpha, 
                     float * A, int lda, 
                     float * B, int ldb,
                     float * C, int ldc)
{
  int i,j,k;
  for(i = 0; i < M; ++i)
  {
    for(j = 0; j < N; ++j)
    {
      register float sum = 0;
      for(k = 0; k < K; ++k)
      {
        sum += alpha * A[i+k*lda] * B[k+j*ldb];
      }
      C[i*ldc+j] += sum;
    }
  }
}


void thread_gemm_beta (int M, int N,
                       float beta,
                       float * C)
{
  int i;
  for (i = 0; i < M*N; ++i)
  {
    C[i] *= beta;
  }
}
