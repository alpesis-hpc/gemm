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

blas_queue_t  queue[MAX_CPU_NUMBER]; 
job_t         job[MAX_CPU_NUMBER];

void queue_init(void);
void queue_run(void);
void queue_exit(void);

#endif
