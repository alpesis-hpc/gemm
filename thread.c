#include "thread.h"

//about pthread:
void sub_pthread_exec(void)
{
    int pthread_pos;    
    for(pthread_pos = 1; pthread_pos < MAX_CPU_NUMBER; pthread_pos++)
    {
        if (thread_status[pthread_pos].status == THREAD_STATUS_SLEEP) 
        {
            pthread_mutex_lock  (&thread_status[pthread_pos].lock);
            thread_status[pthread_pos].status = THREAD_STATUS_WAKEUP;
            pthread_cond_signal(&thread_status[pthread_pos].wakeup);
            pthread_mutex_unlock(&thread_status[pthread_pos].lock);
        }
    }
}

static void* sub_pthread_body(void *arg)
{
    int  pthread_pos = (int)arg;
    pthread_mutex_lock  (&thread_status[pthread_pos].lock);

    while (thread_status[pthread_pos].status == THREAD_STATUS_SLEEP)
    {
        pthread_cond_wait(&thread_status[pthread_pos].wakeup, &thread_status[pthread_pos].lock);
    }
    pthread_mutex_unlock(&thread_status[pthread_pos].lock);

    ((ROUTINE)(queue[pthread_pos].routine))(pthread_pos);
    queue[pthread_pos].assigned = 0;
    thread_status[pthread_pos].status = THREAD_STATUS_SLEEP;
}
