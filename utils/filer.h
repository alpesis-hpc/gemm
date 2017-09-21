#ifndef FILER_H
#define FILER_H

#include <stdio.h>

void file_read(float * A, int buffsize, char * filepath)
{
  int f;

  if((f = fopen(filepath,"rb")) ==-1)
  {
    printf(stderr, "Error: File created error.\n");
  }
  printf("[%s] Read size: %d \r\n",  filepath, fread(A, 4, buffsize, f));

  close(f);
}

#endif
