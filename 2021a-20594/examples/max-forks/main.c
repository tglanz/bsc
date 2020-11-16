#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>

int32_t main(int32_t argc, char** argv) {

    uint32_t counter = 0;

    int32_t pid = fork();

    while (pid > 0) {
        printf("- counter: %d, pid: %d\n", counter, pid);
        counter++;
        pid = fork();
    }

    return 0;
}
