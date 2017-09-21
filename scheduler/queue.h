#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "common.h"

void queue_init (void * routine);
void queue_exit (void);
void queue_run (float * A, float * B, float * C, BLASLONG M, BLASLONG N, BLASLONG K);

void thread_init (void * arg);
void thread_exec (void);
static void * thread_routine (void * arg);


#endif
