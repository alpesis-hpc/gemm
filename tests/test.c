#include "queue.h"
#include "gemm_thread.h"
#include "filer.h"
#include "timer.h"



int main()
{
    float* abuff = malloc(256 * 2400 * 4);
    float* bbuff = malloc(7676 * 2400 * 4);
    float* cbuff = malloc(256 * 7676 * 4);
    file_read(abuff, bbuff);
    printf("Aabuff:%x  Bbbuff:%x Ccbuff:%x \r\n",abuff,bbuff,cbuff);

    struct timespec start, finish;
    double elapsed;
    
    sub_pthread_init();

    double tic = timer();
    sgemm_thread_nn(abuff, bbuff, cbuff, 256, 7676, 2400);
    printf("sgemm_thread_nn elapsed time:%f cbuff[128*7676]:%f\r\n", timer() - tic, cbuff[128*7676]);
    sub_pthread_exit();

    free(abuff);
    free(bbuff);
    free(cbuff);
    
    return 0;
}
