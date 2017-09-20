#include "thread.c"


int main()
{
    float* abuff = malloc(256 * 2400 * 4);
    float* bbuff = malloc(7676 * 2400 * 4);
    float* cbuff = malloc(256 * 7676 * 4);
    int fd_a, fd_b;

    struct timespec start, finish;
    double elapsed;
    
    sub_pthread_init();

    if((fd_a = fopen("./demo_2nd_conv_A","rb")) ==-1)
    {
        printf("A creat file wrong!");
    }
    if((fd_b = fopen("./demo_2nd_conv_B","rb")) ==-1)
    {
        printf("B creat file wrong!");
    }
    printf("A read size:%d \r\n",  fread(abuff, 4, 256 * 2400, fd_a));
    printf("B read size:%d  \r\n", fread(bbuff, 4, 7676 * 2400, fd_b));
    close(fd_a);
    close(fd_b);
    printf("Aabuff:%x  Bbbuff:%x Ccbuff:%x \r\n",abuff,bbuff,cbuff);
    clock_gettime(CLOCK_MONOTONIC, &start);
    {
        sgemm_thread_nn(abuff, bbuff, cbuff, 256, 7676, 2400);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("sgemm_thread_nn elapsed time:%f cbuff[128*7676]:%f\r\n",elapsed,cbuff[128*7676]);

    free(abuff);
    free(bbuff);
    free(cbuff);
    
    sub_pthread_exit();
    return 0;
}
