/**
 * Implementations for set.h header file
 */

#include "set.h"

/**
 * Internal, utility function.
 * 
 * Checks the count and capacity of the set, and according
 * to relevant conditions, re-allocate the set items.
 * 
 * The first condition to re-allocate is when the capacity is 0,
 * well of course, we need some space to add items to the set.
 * In this situation, the capacity will be as defined in the header file
 *  - see SET_INITIAL_CAPACITY
 * 
 * The second condition is when the count and capacity are equal.
 * When this happens, it means we have no more space to add ne items
 * in the future, so we need to perform the re-allocation.
 */
static void set_grow_if_needed(set_t * set){
    if (set){
        if (set->capacity == 0){
            set->items = malloc(sizeof(int) * SET_INITIAL_CAPACITY);
            set->capacity = SET_INITIAL_CAPACITY;
        } else if (set->count == set->capacity){
            set->capacity *= 2;
            set->items = realloc(set->items, set->capacity * sizeof(int));
        }
    }
}

set_t* set_new(){
    /* Allocate a new set object on the heap */
    set_t * set = malloc(sizeof(set_t));

    /**
     * Initialize the set fields.
     * Note;
     *  we could have used calloc,
     *  but (for me at least) it looks more straight forwrad like this
     */
    set->count = 0;
    set->capacity = 0;
    set->items = NULL;

    /* Manage re-allocates as needed */
    set_grow_if_needed(set);

    return set;
}

void set_free(set_t* set){
    /* Check if we got a valid pointer */
    if (set){
        /* And that we have allocated the items before */
        if (set->items){
            free(set->items);
        }

        free(set);
    }
}

int set_add(set_t* set, int value_to_add){
    int is_unique = 1;    /* a flag that tracks the uniqueness */
    unsigned int index; /* iterations index */

    /**
     * Iterate on all the items,
     * assume the value is unique until proven otherwise
     */
    for (index = 0; index < set->count && is_unique; ++index){
        if (set->items[index] == value_to_add){
            is_unique = 0;
        }
    }

    /**
     * If the value is unique, we can add it to the set.
     * We do so, and manage the re-allocates as needed.
     */
    if (is_unique){
        set->items[set->count] = value_to_add;
        set->count++;
        set_grow_if_needed(set);
    }

    return is_unique;
}

/** COULD BE NICE TO HAVE - implementation draft
int set_remove(set_t* set, int value_to_remove){
    int removal = 0;
    unsigned int index;

    for (index = 0; index < set->count; ++index){
        if (removal){
            set->items[index - 1] = set->items[index];
        } else if (set->items[index] == value_to_remove){
            removal = 1;
        }
    }

    if (removal){
        set->count--;
    }

    return removal;
}
*/