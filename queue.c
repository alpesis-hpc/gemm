#include <sys/mman.h>

#include "common.h"
#include "queue.h"
#include "thread.c"

#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)


blas_queue_t queue[MAX_CPU_NUMBER];


void queue_init(void)
{
    int i, j, pthread_pos;
    
    for (i = 0; i < MAX_CPU_NUMBER; i++)
    {
        queue[i].sa       = mmap(NULL, BUFFER_SIZE, MMAP_ACCESS, MMAP_POLICY, -1, 0);
        queue[i].sb       = (void *)(((BLASLONG)(queue[i].sa) + ((SGEMM_P * SGEMM_Q * sizeof(float) + GEMM_ALIGN) & ~GEMM_ALIGN)));
        queue[i].assigned = i + 1;
        queue[i].routine  = inner_thread;
        
        for (j = 0; j < MAX_CPU_NUMBER; j++)
        {
            job[i].working[j][CACHE_LINE_SIZE] = 0;
        }
    }

    for(pthread_pos = 1; pthread_pos < MAX_CPU_NUMBER; pthread_pos++)
    {
        pthread_mutex_init(&thread_status[pthread_pos].lock, NULL);
        pthread_cond_init (&thread_status[pthread_pos].wakeup, NULL);
        thread_status[pthread_pos].status = THREAD_STATUS_SLEEP;
        pthread_create(&blas_threads[pthread_pos], NULL, &sub_pthread_body, (void *)pthread_pos);
    }
}


void queue_run(void)
{
}


void queue_exit(void)
{
  int i;
  for (i = 0; i < MAX_CPU_NUMBER; ++i)
  {
    munmap(queue[i].sa, BUFFER_SIZE);
  }
}
