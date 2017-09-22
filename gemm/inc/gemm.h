#ifndef GEMM_H
#define GEMM_H

void gemm (int ta, int tb, 
           int M, int N, int K, 
           float alpha, 
           float * A, int lda, 
           float * B, int ldb,
           float beta,
           float * C, int ldc);

void gemm_cpu (int ta, int tb, 
               int M, int N, int K, 
               float alpha, 
               float * A, int lda, 
               float * B, int ldb,
               float beta,
               float * C, int ldc);


void gemm_nn (int M, int N, int K, 
              float alpha, 
              float * A, int lda, 
              float * B, int ldb,
              float * C, int ldc);

void gemm_nt (int M, int N, int K, 
              float alpha, 
              float * A, int lda, 
              float * B, int ldb,
              float * C, int ldc);

void gemm_tn (int M, int N, int K, 
              float alpha, 
              float * A, int lda, 
              float * B, int ldb,
              float * C, int ldc);

void gemm_tt (int M, int N, int K, 
              float alpha, 
              float * A, int lda, 
              float * B, int ldb,
              float * C, int ldc);


void gemm_beta (int M, int N, float beta, float * C, int ldc);

#endif 
