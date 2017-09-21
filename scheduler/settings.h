#ifndef SETTINGS_H
#define SETTINGS_H

#include <sched.h>
#include <pthread.h>

#include "common.h"

#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)
 
#define MAX_CPU_NUMBER 8
#define CACHE_LINE_SIZE 8
 
#define THREAD_STATUS_SLEEP 2
#define THREAD_STATUS_WAKEUP 4
  
#define YIELDING sched_yield()

/* ------------------------- */

typedef struct {
   FLOAT * a;
   FLOAT * b;
   FLOAT * c;
   BLASLONG m;
   BLASLONG n;
   BLASLONG k;
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
  void * sa;
  void * sb;
  blas_arg * args;
} scheduler_queue;


typedef struct {
  volatile BLASLONG working[MAX_CPU_NUMBER][CACHE_LINE_SIZE];
} scheduler_job;


typedef struct {
  scheduler_queue * volatile queue __attribute__((aligned(32)));
  pthread_mutex_t lock;
  pthread_cond_t wakeup;
  volatile long status;
} thread_status; 


scheduler_queue QUEUE[MAX_CPU_NUMBER];
scheduler_job JOB[MAX_CPU_NUMBER];
static pthread_t THREADS [MAX_CPU_NUMBER];
static thread_status THREADS_STATUS[MAX_CPU_NUMBER] __attribute__((aligned(128)));

typedef int (*ROUTINE)(BLASLONG);


#endif
