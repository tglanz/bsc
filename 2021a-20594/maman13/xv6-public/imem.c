#include "types.h"
#include "stat.h"
#include "user.h"
#include "fcntl.h"
#define MAX_SECTOR 16522
 
int
main()
{
  char buf[512], buf1[512];
  int fd, i;
  
  for (i = 0; i < 512; i++){
    buf[i] = i;
  }

  fd = open("big.file", O_CREATE | O_WRONLY);
  if(fd < 0){
    printf(2, "big: cannot open big.file for writing\n");
    exit();
  }

  for (i = 0; i < MAX_SECTOR; i++){
    int cc = write(fd, buf, sizeof(buf));
    if (cc < 0){
      printf(2, "failure to write\n");
      exit();
    }
  }

  close(fd);
  fd = open("big.file", O_RDONLY);
  if(fd < 0){
    printf(2, "big: cannot re-open big.file for reading\n");
    exit();
  }

  for(i = 0; i < MAX_SECTOR; i++){
    int cc = read(fd, buf1, sizeof(buf));
    if(cc <= 0){
      printf(2, "big: read error at sector %d\n", i);
      exit();
    }
  
    int j;
    for (j = 0; j < 512; j++){
      if (buf[j] != buf1[j]){
        printf(2, "sector %d: buf[j]!=buf1[j]\n", i);
      }
    }

  }

  exit();
}
