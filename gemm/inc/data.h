#ifndef DATA
#define DATA

#include <stdlib.h>

void random_initializer (float * A, unsigned int count)
{
  int i;
  for (i = 0; i < count; ++i)
  {
    A[i] = rand();
  }
}

#endif
