#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <sys/mman.h>

#include "common.h"
#include "gemm_thread.h"
#include "settings.h"


void divide (BLASLONG M, BLASLONG* range_M)
{
  int dx = M%MAX_CPU_NUMBER;
  int dy = M/MAX_CPU_NUMBER;
  int index = 0;
  int i;
  for(i = 0;i < MAX_CPU_NUMBER + 1; i++)
  {
    range_M[i] = index;
    if(i < dx)
    {
      index = index + dy + 1;
    }
    else
    {
      index = index + dy;
    }
  }
}


static void * pthread_routine (void *arg)
{
  int pthread_pos = (int)arg;
  pthread_mutex_lock  (&THREAD_STATUS[pthread_pos].lock);

  while (THREAD_STATUS[pthread_pos].status == THREAD_STATUS_SLEEP)
  {
    pthread_cond_wait(&THREAD_STATUS[pthread_pos].wakeup, &THREAD_STATUS[pthread_pos].lock);
  }
  pthread_mutex_unlock(&THREAD_STATUS[pthread_pos].lock);

  ((ROUTINE)(QUEUE[pthread_pos].routine))(pthread_pos);
  QUEUE[pthread_pos].assigned = 0;
  THREAD_STATUS[pthread_pos].status = THREAD_STATUS_SLEEP;
}


void pthread_exec(void)
{
  int pthread_pos;    
  for (pthread_pos = 1; pthread_pos < MAX_CPU_NUMBER; pthread_pos++)
  {
    if (THREAD_STATUS[pthread_pos].status == THREAD_STATUS_SLEEP) 
    {
      pthread_mutex_lock (&THREAD_STATUS[pthread_pos].lock);
      THREAD_STATUS[pthread_pos].status = THREAD_STATUS_WAKEUP;
      pthread_cond_signal (&THREAD_STATUS[pthread_pos].wakeup);
      pthread_mutex_unlock (&THREAD_STATUS[pthread_pos].lock);
    }
  }
}

//-------------------------------------------------

void queue_run(float * A, float * B, float * C, BLASLONG M, BLASLONG N, BLASLONG K)
{
  int i;
  sgemm_config (A, B, C, M, N, K);
  divide (BLAS_ARGS.m, range_M);
  divide (BLAS_ARGS.n, range_N);
  pthread_exec();

  ((ROUTINE)(QUEUE[0].routine))(0);
  QUEUE[0].assigned = 0;

  for (i = 0; i < MAX_CPU_NUMBER; i++) while (QUEUE[i].assigned) {YIELDING;};
}


void queue_init(void * routine)
{
  int i, j, pthread_pos;
   
  // init queue and job 
  for (i = 0; i < MAX_CPU_NUMBER; i++)
  {
    QUEUE[i].sa = mmap(NULL, BUFFER_SIZE, MMAP_ACCESS, MMAP_POLICY, -1, 0);
    QUEUE[i].sb = (void *)(((BLASLONG)(QUEUE[i].sa) + ((SGEMM_P * SGEMM_Q * sizeof(float) + GEMM_ALIGN) & ~GEMM_ALIGN)));
    QUEUE[i].assigned = i + 1;
    QUEUE[i].routine = routine;
        
    for (j = 0; j < MAX_CPU_NUMBER; j++)
    {
      JOB[i].working[j][CACHE_LINE_SIZE] = 0;
    }
  }

  // create threads
  for(pthread_pos = 1; pthread_pos < MAX_CPU_NUMBER; pthread_pos++)
  {
    pthread_mutex_init(&THREAD_STATUS[pthread_pos].lock, NULL);
    pthread_cond_init (&THREAD_STATUS[pthread_pos].wakeup, NULL);
    THREAD_STATUS[pthread_pos].status = THREAD_STATUS_SLEEP;
    pthread_create(&BLAS_THREADS[pthread_pos], NULL, &pthread_routine, (void *)pthread_pos);
  }
}


void queue_exit(void)
{
  int i;
  for (i = 0; i < MAX_CPU_NUMBER; i++)
  {
    munmap(QUEUE[i].sa, BUFFER_SIZE);
  }
}

#endif
