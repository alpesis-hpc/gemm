#ifndef GEMM_THREAD_H
#define GEMM_THREAD_H

void divide(BLASLONG M, BLASLONG* range_M)
{
    int dx = M%MAX_CPU_NUMBER;
    int dy = M/MAX_CPU_NUMBER;
    int index = 0;
    int i;
    for(i = 0;i < MAX_CPU_NUMBER + 1; i++)
    {
        range_M[i] = index;
        if(i < dx)
        {
            index = index + dy + 1;
        }
        else
        {
            index = index + dy;
        }
    }
}


void sgemm_thread_nn(float* A, float* B, float* C, BLASLONG M, BLASLONG N, BLASLONG K)
{
    int i;
    execute_arg.a        = B;
    execute_arg.b        = A;
    execute_arg.c        = C;
    execute_arg.m        = N;
    execute_arg.n        = M;
    execute_arg.k        = K;

    divide(execute_arg.m, range_M);
    divide(execute_arg.n, range_N);

    sub_pthread_exec();
    
    ((ROUTINE)(queue[0].routine))(0);
    queue[0].assigned = 0;

    for (i = 0; i < MAX_CPU_NUMBER; i++) while (queue[i].assigned) {YIELDING;};
}


#endif
