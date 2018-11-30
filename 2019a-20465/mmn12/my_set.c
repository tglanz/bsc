/**
 * my_set.c
 * 
 * The application reads indefinite amount of integers from stdin,
 * adds them into a set object, and print the set.
 */

#include <stdio.h>

/* The set api */
#include "set.h"

/**
 * Given a set, print it's information - Count, Capacity and Items
 */
static void print_set(set_t* const set);

/**
 * Acquire integers from stdin and initialize a set with those values
 * 
 * Returns
 *  set_t* - defined in set.h
 */
static set_t* get_set();

/**
 * Main entry to the application
 */
int main(int argc, char ** argv){
    /* Just do what we need; get, print delete */
    set_t* set = get_set();
    print_set(set);
    set_free(set);
    return 0;
}

static set_t* get_set() {
    int input_number = 0;
    int ret_val = 0;

    /**
     * Use the set api to create a new set object on the heap.
     * Don't forget to clean it up (!)
     */
    set_t* set = set_new();

    /* As long as we havn't reached the eof token, read integers */
    while ((ret_val = scanf("%d", &input_number)) != EOF){
        /* Use the set api to add the number */
        set_add(set, input_number);
    }

    return set;
}

static void print_set(set_t* const set) {

    /**
     * Print the set fields
     * Iterate the items and print them as well
     */

    unsigned int index = 0;

    if (!set){
        printf("Unable to print null set...\n");
    } else {
        printf("-------- Set --------\n");
        printf("Count: %d\n", set->count);
        printf("Capacity: %d\n", set->capacity);
        if (set->count > 0){
            printf("Items:\n");
            for (index = 0; index < set->count; ++index){
                printf("  %d. %d\n", index, set->items[index]);
            }
        }
        printf("---------------------\n");
    }
}