#include "gemm.h"


void gemm (int ta, int tb, 
           int M, int N, int K, 
           float alpha, 
           float * A, int lda, 
           float * B, int ldb,
           float beta,
           float * C, int ldc)
{
  gemm_cpu (ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}


void gemm_cpu (int ta, int tb, 
               int M, int N, int K, 
               float alpha, 
               float * A, int lda, 
               float * B, int ldb,
               float beta,
               float * C, int ldc)
{
  gemm_beta (M, N, beta, C, ldc);

  if      (!ta && !tb)      gemm_nn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
  else if (ta && !tb)       gemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
  else if (!ta && tb)       gemm_nt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
  else                      gemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}


void gemm_nn(int M, int N, int K, 
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


void gemm_nt(int M, int N, int K, 
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


void gemm_tn(int M, int N, int K, 
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


void gemm_tt(int M, int N, int K, 
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


void gemm_beta (int M, int N,
                float beta,
                float * C, int ldc)
{
  int i, j;
  for(i = 0; i < M; ++i)
  {
    for(j = 0; j < N; ++j)
    {
      C[i*ldc + j] *= beta;
    }
  }
}
