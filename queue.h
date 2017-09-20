#ifndef TYPES_H
#define TYPES_H

#include "common.h"

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


void queue_init(void);
void queue_run(void);
void queue_exit(void);

#endif
