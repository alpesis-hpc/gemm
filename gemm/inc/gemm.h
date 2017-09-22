#ifndef GEMM_H
#define GEMM_H

void gemm (int TA, int TB, 
           int M, int N, int K, 
           float ALPHA, 
           float *A, int lda, 
           float *B, int ldb,
           float BETA,
           float *C, int ldc);

void gemm_cpu (int TA, int TB, 
               int M, int N, int K, 
               float ALPHA, 
               float *A, int lda, 
               float *B, int ldb,
               float BETA,
               float *C, int ldc);


void gemm_nn (int M, int N, int K, 
              float ALPHA, 
              float *A, int lda, 
              float *B, int ldb,
              float *C, int ldc);

void gemm_nt (int M, int N, int K, 
              float ALPHA, 
              float *A, int lda, 
              float *B, int ldb,
              float *C, int ldc);

void gemm_tn (int M, int N, int K, 
              float ALPHA, 
              float *A, int lda, 
              float *B, int ldb,
              float *C, int ldc);

void gemm_tt (int M, int N, int K, 
              float ALPHA, 
              float *A, int lda, 
              float *B, int ldb,
              float *C, int ldc);


void gemm_beta (int M, int N, float beta, float * C, int ldc);

#endif 
