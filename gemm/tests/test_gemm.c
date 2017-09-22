/*
 *   complex     -> double -> zgemm
                 -> single -> cgemm
     not complex -> double -> dgemm
                 -> single -> sgemm
 */

/* ----------------------------------- */


/* ----------------------------------- */


#include <stdio.h>
#include <stdlib.h>
#ifdef __CYGWIN32__
#include <sys/time.h>
#endif
#include "common.h"


#undef GEMM

#ifndef COMPLEX

#ifdef DOUBLE
#define GEMM   BLASFUNC(dgemm)
#else
#define GEMM   BLASFUNC(sgemm)
#endif

#else

#ifdef DOUBLE
#define GEMM   BLASFUNC(zgemm)
#else
#define GEMM   BLASFUNC(cgemm)
#endif

#endif

#if defined(__WIN32__) || defined(__WIN64__)

#ifndef DELTA_EPOCH_IN_MICROSECS
#define DELTA_EPOCH_IN_MICROSECS 11644473600000000ULL
#endif

int gettimeofday(struct timeval *tv, void *tz){

  FILETIME ft;
  unsigned __int64 tmpres = 0;
  static int tzflag;

  if (NULL != tv)
    {
      GetSystemTimeAsFileTime(&ft);

      tmpres |= ft.dwHighDateTime;
      tmpres <<= 32;
      tmpres |= ft.dwLowDateTime;

      /*converting file time to unix epoch*/
      tmpres /= 10;  /*convert into microseconds*/
      tmpres -= DELTA_EPOCH_IN_MICROSECS;
      tv->tv_sec = (long)(tmpres / 1000000UL);
      tv->tv_usec = (long)(tmpres % 1000000UL);
    }

  return 0;
}

#endif

#if !defined(__WIN32__) && !defined(__WIN64__) && !defined(__CYGWIN32__) && 0

static void *huge_malloc(BLASLONG size){
  int shmid;
  void *address;

#ifndef SHM_HUGETLB
#define SHM_HUGETLB 04000
#endif

  if ((shmid =shmget(IPC_PRIVATE,
		     (size + HUGE_PAGESIZE) & ~(HUGE_PAGESIZE - 1),
		     SHM_HUGETLB | IPC_CREAT |0600)) < 0) {
    printf( "Memory allocation failed(shmget).\n");
    exit(1);
  }

  address = shmat(shmid, NULL, SHM_RND);

  if ((BLASLONG)address == -1){
    printf( "Memory allocation failed(shmat).\n");
    exit(1);
  }

  shmctl(shmid, IPC_RMID, 0);

  return address;
}

#define malloc huge_malloc

#endif

int main(int argc, char *argv[])
{

  FLOAT *a, *b, *c;
  FLOAT alpha[] = {1.0, 1.0};
  FLOAT beta [] = {0.0, 0.0};
  char trans='N';
  blasint m, n, i, j;
  int loops = 1;
  int has_param_n=0;
  int l;
  char *p;

  int from =   1;
  int to   = 200;
  int step =   1;

  struct timeval start, stop;
  double time1,timeg;

  argc--;argv++;
  if (argc > 0) { from     = atol(*argv);		argc--; argv++;}
  if (argc > 0) { to       = MAX(atol(*argv), from);	argc--; argv++;}
  if (argc > 0) { step     = atol(*argv);		argc--; argv++;}

  if ((p = getenv("OPENBLAS_TRANS")))  trans=*p;
  fprintf(stderr, "From : %3d  To : %3d Step=%d : Trans=%c\n", from, to, step, trans);

  if (( a = (FLOAT *)malloc(sizeof(FLOAT) * to * to * COMPSIZE)) == NULL)
  {
    fprintf(stderr,"Out of Memory!!\n");exit(1);
  }

  if (( b = (FLOAT *)malloc(sizeof(FLOAT) * to * to * COMPSIZE)) == NULL)
  {
    fprintf(stderr,"Out of Memory!!\n");exit(1);
  }

  if (( c = (FLOAT *)malloc(sizeof(FLOAT) * to * to * COMPSIZE)) == NULL)
  {
    fprintf(stderr,"Out of Memory!!\n");exit(1);
  }

  p = getenv("OPENBLAS_LOOPS");
  if ( p != NULL ) loops = atoi(p);
  if ((p = getenv("OPENBLAS_PARAM_N"))) 
  {
          n = atoi(p);
	  has_param_n=1;	  
  }

#ifdef linux
  srandom(getpid());
#endif

  // assign a, b, c with random values 
	for(j = 0; j < to; j++)
        {
      		for(i = 0; i < to * COMPSIZE; i++)
                {
			a[i + j * to * COMPSIZE] = ((FLOAT) rand() / (FLOAT) RAND_MAX) - 0.5;
			b[i + j * to * COMPSIZE] = ((FLOAT) rand() / (FLOAT) RAND_MAX) - 0.5;
			c[i + j * to * COMPSIZE] = ((FLOAT) rand() / (FLOAT) RAND_MAX) - 0.5;
      		}
    	}



  fprintf(stderr, "   SIZE          Flops          Time\n");
  for(m = from; m <= to; m += step)
  {
    timeg=0;
    if ( has_param_n == 1 && n <= m )
    	n=n;
    else
    	n=m;
    fprintf(stderr, " %6dx%d : ", (int)m, (int)n);
    gettimeofday( &start, (struct timezone *)0);

    for (l=0; l<loops; l++)
    {
      GEMM (&trans, &trans, &m, &n, &m, alpha, a, &m, b, &m, beta, c, &m );
    }
   gettimeofday( &stop, (struct timezone *)0);
   time1 = (double)(stop.tv_sec - start.tv_sec) + (double)((stop.tv_usec - start.tv_usec)) * 1.e-6;

    timeg = time1/loops;
    fprintf(stderr,
	    " %10.2f MFlops %10.6f sec\n",
	    COMPSIZE * COMPSIZE * 2. * (double)m * (double)m * (double)n / timeg * 1.e-6, time1);
  }

  return 0;
}
