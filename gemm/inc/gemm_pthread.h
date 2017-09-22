#ifndef GEMM_PTHREAD_H
#define GEMM_PTHREAD_H

void thread_gemm (int ta, int tb, 
                  int M, int N, int K, 
                  float alpha, 
                  float * A, int lda, 
                  float * B, int ldb,
                  float beta,
                  float * C, int ldc);

void thread_gemm_cpu (int ta, int tb, 
                      int M, int N, int K, 
                      float alpha, 
                      float * A, int lda, 
                      float * B, int ldb,
                      float beta,
                      float * C, int ldc);


void thread_gemm_nn (int M, int N, int K, 
                     float alpha, 
                     float * A, int lda, 
                     float * B, int ldb,
                     float * C, int ldc);

void thread_gemm_nt (int M, int N, int K, 
                     float alpha, 
                     float * A, int lda, 
                     float * B, int ldb,
                     float * C, int ldc);

void thread_gemm_tn (int M, int N, int K, 
                     float alpha, 
                     float * A, int lda, 
                     float * B, int ldb,
                     float * C, int ldc);

void thread_gemm_tt (int M, int N, int K, 
                     float alpha, 
                     float * A, int lda, 
                     float * B, int ldb,
                     float * C, int ldc);


void thread_gemm_beta (int M, int N, float beta, float * C, int ldc);

#endif 
