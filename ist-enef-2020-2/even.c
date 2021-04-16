#include <stdio.h>
#include <stdlib.h>

int main ()
{
  for(int i=0; i<32; i++)
    i%2 == 0? printf("%d", i) : printf("\n");
  exit(0);
}
