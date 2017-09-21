#include <stdlib.h>
#include <sys/mman.h>
#include <pthread.h>

#include "settings.h"
#include "queue.h"


void queue_run(float * A, float * B, float * C, BLASLONG M, BLASLONG N, BLASLONG K)
{
  int i;
  sgemm_config (A, B, C, M, N, K);
  thread_exec();

  ((ROUTINE)(QUEUE[0].routine))(0);
  QUEUE[0].assigned = 0;

  for (i = 0; i < MAX_CPU_NUMBER; i++) while (QUEUE[i].assigned) {YIELDING;};
}


void queue_exit(void)
{
  int i;
  for (i = 0; i < MAX_CPU_NUMBER; i++)
  {
    munmap(QUEUE[i].sa, BUFFER_SIZE);
  }
}


void queue_init(void * routine)
{
  int i, j;
   
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
  for(i = 1; i < MAX_CPU_NUMBER; i++)
  {
    thread_init (i);
  }
}


void thread_init (void * arg)
{
  int i = (int)arg;

  // init thread i
  pthread_mutex_init (&THREADS_STATUS[i].lock, NULL);
  pthread_cond_init (&THREADS_STATUS[i].wakeup, NULL);
  THREADS_STATUS[i].status = THREAD_STATUS_SLEEP;

  // create thread
  pthread_create (&THREADS[i], NULL, &thread_routine, (void *)i);
}


void thread_exec (void)
{
  int i;    
  for (i = 1; i < MAX_CPU_NUMBER; i++)
  {
    if (THREADS_STATUS[i].status == THREAD_STATUS_SLEEP) 
    {
      pthread_mutex_lock (&THREADS_STATUS[i].lock);
      THREADS_STATUS[i].status = THREAD_STATUS_WAKEUP;
      pthread_cond_signal (&THREADS_STATUS[i].wakeup);
      pthread_mutex_unlock (&THREADS_STATUS[i].lock);
    }
  }
}


static void * thread_routine (void * arg)
{
  int i = (int)arg;

  pthread_mutex_lock (&THREADS_STATUS[i].lock);
  while (THREADS_STATUS[i].status == THREAD_STATUS_SLEEP)
  {
    pthread_cond_wait (&THREADS_STATUS[i].wakeup, &THREADS_STATUS[i].lock);
  }
  pthread_mutex_unlock (&THREADS_STATUS[i].lock);

  ((ROUTINE)(QUEUE[i].routine))(i);
  QUEUE[i].assigned = 0;
  THREADS_STATUS[i].status = THREAD_STATUS_SLEEP;
}
