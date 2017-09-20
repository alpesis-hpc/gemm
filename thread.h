#ifndef THREAD_H
#define THREAD_H

#include "queue.h"

typedef struct {
     blas_queue_t * volatile queue  __attribute__((aligned(32)));
     volatile long status;
     pthread_mutex_t lock;
     pthread_cond_t wakeup;
} thread_status_t;

static thread_status_t thread_status[MAX_CPU_NUMBER] __attribute__((aligned(128)));
typedef int (*ROUTINE)(BLASLONG);


void sub_pthread_exec(void);
static void* sub_pthread_body(void *arg);

#endif
