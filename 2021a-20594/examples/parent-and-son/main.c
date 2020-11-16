#include <stdio.h>

#include <stdbool.h>
#include <unistd.h>

bool isChild(int pid) {
    return pid == 0; 
}

int main(int argc, char** argv){ 
    int pid = fork();

    if (isChild(pid)) {
        printf("i am the child\n");
    } else {
        printf("i am the parent\n");
    }

    return 0;
}

