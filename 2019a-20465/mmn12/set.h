/**
 * set.h
 * 
 * Definitions for the set api
 * 
 * This file contains the definition of the set_t struct,
 * which is the object that the api works on.
 * 
 * Design Note:
 * The struct design is a single approach, another approach is
 * to keep the data flatten in a single, dynamicaly allocated array.
 * The array will contain to values in offsets 0 and 1 which will act
 * as the count and capacity, while the other values will act as the items.
 * For example - the array [4, 6, 3, 5, 2, 1]
 *  will indicate a set of count 4, capacity 6 and items [3, 5, 2, 1]
 * The logic remains the same, but the functions will recieve and return
 * pointers to the array offsetted by 2. (i.e return &set[2]),
 * and the count and capacity will be accessed by set[-2] and set[-1] accordingly.
 * I can't really see a practical difference between the two;
 */
#ifndef SET_H
#define SET_H

#include <stdio.h>
#include <stdlib.h>

/* The capacity for newly created sets */
#define SET_INITIAL_CAPACITY (4)

/* Define the set struct */
typedef struct {
    int capacity;
    int count;
    int* items;
} set_t;

/**
 * Initializes a new set.
 * Use set_free to free the allocated memory
 * 
 * Returns
 *  set_t* - a new set instance
 */
set_t* set_new();

/**
 * Free the memory allocated for the given set
 * 
 * Arguments
 *  set: set_t* - a pointer to a set object
 */
void set_free(set_t* set);

/**
 * Adds a new value to the set.
 * If the value already exists, it won't be added.
 * 
 * Arguments
 *  set: set_t* - the set to add the value to
 *  value_to_add: int - the value to add to the set
 * 
 * Returns
 *  a flag to inicate whether the value was added or not
 */
int set_add(set_t* set, int value_to_add);

/** COULD BE NICE TO HAVE - implementation draft in set.c
int set_remove(set_t* set, int value);
*/

#endif /* SET_H */