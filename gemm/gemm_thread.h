#ifndef GEMM_THREAD_H
#define GEMM_THREAD_H

#include "settings.h"


void divide (BLASLONG M, BLASLONG * range_M)
{
  int dx = M % MAX_CPU_NUMBER;
  int dy = M / MAX_CPU_NUMBER;
  int index = 0;
  int i;
  for (i = 0; i < MAX_CPU_NUMBER + 1; i++)
  {
    range_M[i] = index;
    if (i < dx) index = index + dy + 1;
    else        index = index + dy;
  }
}


static inline int sgemm_nn_thread (BLASLONG mypos)
{
  BLASLONG m_from;
  BLASLONG m_to;
  BLASLONG n_from;
  BLASLONG n_to;
  BLASLONG N_from;
  BLASLONG N_to;

  BLASLONG lda;
  BLASLONG ldb;
  BLASLONG ldc;

  BLASLONG ls;
  BLASLONG min_l;
  BLASLONG jjs;
  BLASLONG min_jj;
  BLASLONG is;
  BLASLONG min_i;
  BLASLONG div_n;
  BLASLONG i;
  BLASLONG current;

  FLOAT * a;
  FLOAT * b;
  FLOAT * c;
  FLOAT * alpha;
  FLOAT * beta;
  BLASLONG m;
  BLASLONG n;
  BLASLONG k;
    
  FLOAT ALP = 1;
  FLOAT BET = 0;

  FLOAT * sa = QUEUE[mypos].sa;
  FLOAT * buffer = QUEUE[mypos].sb;
    
  a = BLAS_ARGS.a;
  b = BLAS_ARGS.b;
  c = BLAS_ARGS.c;
  m = BLAS_ARGS.m;
  n = BLAS_ARGS.n;
  k = BLAS_ARGS.k;
  lda = m;
  ldb = k;
  ldc = m;
  alpha = &ALP;
  beta  = &BET;
    
  m_from = range_M[mypos + 0];
  m_to   = range_M[mypos + 1];
  n_from = range_N[mypos + 0];
  n_to   = range_N[mypos + 1];
  N_from = range_N[0];
  N_to   = range_N[MAX_CPU_NUMBER];

  if (beta[0] == 0) BETA_OPERATION (m_from, m_to, N_from, N_to, beta, c, ldc);
  
  for(ls = 0; ls < k; ls += min_l)
  {
    min_l = k - ls;
    if      (min_l >= GEMM_Q * 2)   min_l = GEMM_Q;
    else if (min_l > GEMM_Q)        min_l = (min_l + 1) / 2;

    min_i = m_to - m_from;
    if      (min_i >= GEMM_P * 2)   min_i = GEMM_P;
    else if (min_i > GEMM_P)        min_i = ((min_i / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
 
    ICOPY_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);

    /* Make sure if no one is using buffer */
    for (i = 0; i < MAX_CPU_NUMBER; i++) while(JOB[mypos].working[i][CACHE_LINE_SIZE]) {YIELDING;};

    for(jjs = n_from; jjs < n_to; jjs += min_jj)
    {
      min_jj = n_to - jjs;
      if      (min_jj >= 3*GEMM_UNROLL_N)  min_jj = 3*GEMM_UNROLL_N;
      else if (min_jj >= 2*GEMM_UNROLL_N)  min_jj = 2*GEMM_UNROLL_N;
      else if (min_jj > GEMM_UNROLL_N)     min_jj = GEMM_UNROLL_N;

      OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs, buffer + min_l * (jjs - n_from));
      KERNEL_OPERATION(min_i, min_jj, min_l, alpha, sa, buffer + min_l * (jjs - n_from), c, ldc, m_from, jjs);
    }

    for (i = 0; i < MAX_CPU_NUMBER; i++) JOB[mypos].working[i][CACHE_LINE_SIZE] = (BLASLONG)buffer;
    WMB;

    current = mypos;
    do
    {
      current++; if (current >= MAX_CPU_NUMBER) current = 0;
      if (current != mypos)
      {
        /* thread has to wait */
        while(JOB[current].working[mypos][CACHE_LINE_SIZE] == 0) {YIELDING;};

        KERNEL_OPERATION(min_i, 
                         range_N[current + 1]  - range_N[current], 
                         min_l, 
                         alpha,
                         sa, 
                         (FLOAT *)JOB[current].working[mypos][CACHE_LINE_SIZE],
                         c, 
                         ldc, 
                         m_from, 
                         range_N[current]);
       }
       if (m_to - m_from == min_i) JOB[current].working[mypos][CACHE_LINE_SIZE] &= 0;
     } while (current != mypos);

     for(is = m_from + min_i; is < m_to; is += min_i)
     {
       min_i = m_to - is;

       if      (min_i >= GEMM_P * 2)   min_i = GEMM_P;
       else if (min_i > GEMM_P)        min_i = (((min_i + 1) / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;

       ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa);

       current = mypos;
       do
       {
         KERNEL_OPERATION(min_i, range_N[current + 1] - range_N[current], min_l, alpha,
                    sa, (FLOAT *)JOB[current].working[mypos][CACHE_LINE_SIZE], c, ldc, is, range_N[current]);

         if (is + min_i >= m_to)
         {
           /* Thread doesn't need this buffer any more */
           JOB[current].working[mypos][CACHE_LINE_SIZE] &= 0;
           WMB;
         }

          current ++; if (current >= MAX_CPU_NUMBER) current = 0;
        } while (current != mypos);
      }
  }
  for (i = 0; i < MAX_CPU_NUMBER; i++)  while (JOB[mypos].working[i][CACHE_LINE_SIZE] ) {YIELDING;};
}


void sgemm_config (float * A, float * B, float * C, BLASLONG M, BLASLONG N, BLASLONG K)
{
  BLAS_ARGS.a = B;
  BLAS_ARGS.b = A;
  BLAS_ARGS.c = C;
  BLAS_ARGS.m = N;
  BLAS_ARGS.n = M;
  BLAS_ARGS.k = K;

  divide (BLAS_ARGS.m, range_M);
  divide (BLAS_ARGS.n, range_N);
}


#endif
