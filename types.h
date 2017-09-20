#ifndef TYPES_H
#define TYPES_H

#include <sys/mman.h>

#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)

typedef struct {
   volatile BLASLONG working[MAX_CPU_NUMBER][CACHE_LINE_SIZE];
} job_t;


typedef struct {
  FLOAT *a, *b, *c;
  BLASLONG m, n, k;
  BLASLONG nthreads;
  void *common;
} blas_arg_t;

typedef struct blas_queue {
  void *routine;
  volatile int assigned;
  void *sa, *sb;
  blas_arg_t *args;
} blas_queue_t;


typedef struct {
    blas_queue_t * volatile queue  __attribute__((aligned(32)));
    volatile long status;
    pthread_mutex_t lock;
    pthread_cond_t wakeup;
} thread_status_t;


#endif
