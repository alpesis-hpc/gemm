#ifndef SETTINGS_H
#define SETTINGS_H

#include "common.h"

#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)
 
#define MAX_CPU_NUMBER 8
 
#define THREAD_STATUS_SLEEP 2
#define THREAD_STATUS_WAKEUP 4
  
#define YIELDING sched_yield()

/* ------------------------- */

typedef struct {
   FLOAT *a, *b, *c;
   BLASLONG m, n, k;
   BLASLONG nthreads;
   void *common;
} blas_arg;
 
blas_arg BLAS_ARGS;
BLASLONG range_M[MAX_CPU_NUMBER + 1];
BLASLONG range_N[MAX_CPU_NUMBER + 1];


/* ------------------------- */

typedef struct blas_queue {
  void * routine;
  volatile int assigned;
  void * sa, * sb;
  blas_arg * args;
} queue;


typedef struct {
  queue * volatile queue  __attribute__((aligned(32)));
  volatile long status;
  pthread_mutex_t lock;
  pthread_cond_t wakeup;
} thread_status_t; 


typedef struct {
  volatile BLASLONG working[MAX_CPU_NUMBER][CACHE_LINE_SIZE];
} job;


static thread_status_t THREAD_STATUS[MAX_CPU_NUMBER] __attribute__((aligned(128)));
static pthread_t BLAS_THREADS [MAX_CPU_NUMBER];

queue QUEUE[MAX_CPU_NUMBER];
job JOB[MAX_CPU_NUMBER];


typedef int (*ROUTINE)(BLASLONG);


#endif
