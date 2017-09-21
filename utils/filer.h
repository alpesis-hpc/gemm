#ifndef FILER_H
#define FILER_H

void file_read(float * a, float * b)
{
  int fd_a, fd_b;

  if((fd_a = fopen("./data/demo_2nd_conv_A","rb")) ==-1)
  {
    printf("A creat file wrong!");
  }

  if((fd_b = fopen("./data/demo_2nd_conv_B","rb")) ==-1)
  {
    printf("B creat file wrong!");
  }
  printf("A read size:%d \r\n",  fread(a, 4, 256 * 2400, fd_a));
  printf("B read size:%d  \r\n", fread(b, 4, 7676 * 2400, fd_b));
  close(fd_a);
  close(fd_b);
}

#endif
