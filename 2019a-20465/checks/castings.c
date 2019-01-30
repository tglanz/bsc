#include <stdio.h>

int main(int argc, char ** argv){
    int some_int;
    char some_char;

    printf("start\n");

    some_char = 'c';
    some_int = some_char;
    printf("to-int.implicit = %d\n", some_int);

    some_char = 'c';
    some_int = (int)some_char;
    printf("to-int.explicit = %d\n", some_int);

    some_int = 256 + 97;
    some_char = some_int;
    printf("to-char.implicit = %c\n", some_char);

    some_int = 256 + 97;
    some_char = (char)some_int;
    printf("to-char.explicit = %c\n", some_char);

    return 0;
}
