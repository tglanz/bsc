/*
 * Simulate base 2 addition
 */

#include <stdio.h>

/**
 * Definitions
 *
 * EXIT_CODE_OK - An exit code, indicates that the program finished execution successfully
 * MAX_BITS - The maximum number of bits we support, according to info
 *            the input numbers have up to 6 decimal digits, it means
 *            no more than 20 bits 
 * PARITY - Base two
 */
#define EXIT_CODE_OK (0)                                
#define MAX_BITS (20)
#define PARITY (2)

/**
 * Prints a binary computation to stdout
 * Assumes that the arrays have length of MAX_BITS
 */
void print_binary_computation(
        unsigned char * a_bits,
        unsigned char * b_bits,
        unsigned char * result_bits);

/**
 * Prints a decimal computation to stdout
 */
void print_decimal_computation(unsigned int a, unsigned int b, unsigned int result);

/**
 * Reads the input from stdin
 */
void read_input(unsigned int * a, unsigned int * b);

/**
 * Prints a given bit array to stdout
 */
void print_array(unsigned char * array, unsigned int count);

/**
 * The function we actually care about.
 * Simulate the addition operation in base 2
 *
 * To get the rightmost bit, we use the modulu operation
 * Given two bits, a and b, we notice that the base two result is (a+b)%2 and carry is (a+b)/2
 *
 * Prints the binary computation and returns the result as an integer
 */
unsigned int my_add(unsigned int a, unsigned int b);

/*
 * Program entry point
 *
 * Read two number from stdin
 * Simulate base 2 addition
 * Print base 2 computation
 * Print base 10 computation
 */
int main(int argc, char ** argv){
    unsigned int a;         /* the first operand */
    unsigned int b;         /* the second operand */
    unsigned int result;    /* the result of the addition */

    read_input(&a, &b);
    printf("\n");

    result = my_add(a, b);
    print_decimal_computation(a, b, result);
    
    return EXIT_CODE_OK;
}

void print_decimal_computation(
    unsigned int a, unsigned int b, unsigned int result){

    unsigned int idx;
    printf("\nDecimal Compuation\n");
    printf("  %6d\n", a);
    printf("+ %6d\n", b);
    for (idx = 0; idx < 8; ++idx){
        printf("-");
    }
    printf("\n= %6d\n", result);
}


void print_binary_computation(
        unsigned char * a_bits,
        unsigned char * b_bits,
        unsigned char * result_bits){

    /* Just print, using the print_array utility function */

    unsigned int idx = 0;

    printf("Binary Computation\n  ");
    print_array(a_bits, MAX_BITS);
    printf("\n+ ");
    print_array(b_bits, MAX_BITS);
    printf("\n");
    for (idx = 0; idx < 6 + MAX_BITS; ++idx){
        printf("-");
    }
    printf("\n= ");
    print_array(result_bits, MAX_BITS);
    printf("\n");
}

void read_input(unsigned int * a, unsigned int * b){
    printf("Please enter values for operands a and b below\n");
    printf(" > a: ");
    scanf("%u", a);
    printf("%u\n", *a);
    printf(" > b: ");
    scanf("%u", b);
    printf("%u\n", *b);
}

void print_array(unsigned char * array, unsigned int count){
    /* Iterate over the array according to `count`, print each element */
    unsigned int idx;
    for (idx = 0; idx < count; ++idx){
        printf("%d", array[count - idx - 1]);
        if ((count - idx - 1) % 4 == 0){
            printf(" ");
        }
    }
}

unsigned int my_add(unsigned int a, unsigned int b){

    unsigned int idx;       /* Used for looping */
    unsigned int sum;       /* Holds sums of corresponding bits */
    unsigned int result;    /* The addition result */
    unsigned int power;     /* Base 2 power,
                               used to reconstruct the result from it's bits */

    unsigned char carry = 0;    /* The carry for the current addition,
                                    acquired in the previous iteration */

    /* The bits for each operand and result */
    unsigned char a_bits[MAX_BITS];
    unsigned char b_bits[MAX_BITS];
    unsigned char result_bits[MAX_BITS];

    /* Iterate according to the number of bits we expect to be in the result.
     * In case both operands are exhausted, we rely on 0 additions */
    for (idx = 0; idx < MAX_BITS; ++idx){

        a_bits[idx] = a % PARITY; 
        b_bits[idx] = b % PARITY;

        /*
         * Didn't fully understand from the task if this operation is what we needed
         * to do, or by if conditions; such as
         * if (a_bits[idx] == b_bits[idx] && a_bits[idx] == 1){
         *     sum = 0;
         *     carry = 1;
         * } etc.. 
         */
        sum = carry + a_bits[idx] + b_bits[idx];
        carry = sum / PARITY; 
        result_bits[idx] = sum % PARITY; 

        a /= PARITY;
        b /= PARITY;
    }

    print_binary_computation(a_bits, b_bits, result_bits);

    /* Reconstruct the result from its bit's
     * bit i has the value of 2^i in base 10 */
    result = 0;
    power = 1;
    for (idx = 0; idx < MAX_BITS; ++idx){
        result += result_bits[idx] * power; 
        power *= PARITY;
    }

    return result;
}
