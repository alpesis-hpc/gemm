#include "common.h"
#include "queue.h"

// queue / job
BLASLONG      range_M[MAX_CPU_NUMBER + 1];
BLASLONG      range_N[MAX_CPU_NUMBER + 1];
blas_arg_t    execute_arg;

static pthread_t     blas_threads [MAX_CPU_NUMBER];

blas_queue_t  queue[MAX_CPU_NUMBER];


static void pthread_bind(int cpu);

//about instance matrix compute:
static int inner_thread(BLASLONG mypos);
void divide(BLASLONG M, BLASLONG* range_M);
void sgemm_thread_nn(float* A, float* B, float* C, BLASLONG M, BLASLONG N, BLASLONG K);
