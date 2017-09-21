#include "queue.h"
#include "gemm_thread.h"
#include "filer.h"
#include "timer.h"


int main()
{
    float * A = malloc(256 * 2400 * 4);
    float * B = malloc(7676 * 2400 * 4);
    float * C = malloc(256 * 7676 * 4);
    file_read(A, 256*2400, "./data/demo_2nd_conv_A");
    file_read(B, 7676*2400, "./data/demo_2nd_conv_B");
    printf("A: %x  B: %x C: %x \r\n", A, B, C);

    sub_pthread_init();
    double tic = timer();
    sgemm_thread_nn(A, B, C, 256, 7676, 2400);
    printf("(gemm_nn) elapsed:%f, C[128*7676]: %f\r\n", timer() - tic, C[128*7676]);
    sub_pthread_exit();

    free(A);
    free(B);
    free(C);
    
    return 0;
}
