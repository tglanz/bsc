#include <stdio.h>

/**
 * Definitions
 * 
 * MAX_STRING_LENGTH - according to the task, number is characters is less than 80
 * EXIT_CODE_OK - exit code, indicates that the program run successfuly
 */
#define MAX_STRING_LENGTH (80)
#define EXIT_CODE_OK (0)

/**
 * Given a null terminated string, return it's length.
 * Examples:
 *   - "abcd\0" -> 4
 *   - "aa  bb  cc\0" -> 10
 */
unsigned int get_length(char * string);

/**
 * Given a character, determines if its a whitespace.
 * Note - whitespace characters according to our needs
 * Examples:
 *   - ' ' -> 1
 *   - '\t' -> 1
 *   - 'a' -> 0
 */
int is_whitespace(char character);

/**
 * The function we actually care about.
 * Given a null terminated string, determine if it's a palindrome.
 * The function ignores whitespaces (see is_whitespace).
 * 
 * The logic behind this function is to traverse the string both from left
 * and from right, and comparing each pair of non-whitespace characters.
 * 
 * By default, any string is a palindrome until proven otherwise;
 * For example, the empty string is a palindrome.
 * 
 * Case insensitive.
 * 
 * Examples:
 *   - "a b c b a" -> 1
 *   - "never odd or even" -> 1
 *   - "abba" -> 1
 *   - "Abba" -> 0
 *   - "a bc de" -> 0
 */
int palindrome(char s[]);

/**
 * The entry point.
 * Read a line, check if it's a palindrome, and print the result
 */
int main(int argc, char ** argv){
    unsigned int idx = 0;
    char string[MAX_STRING_LENGTH];
    int is_palindrome = 0;

    printf("Enter a string to check if it's a palindrome\n> ");
    fgets(string, MAX_STRING_LENGTH, stdin);

    is_palindrome = palindrome(string);

    /**
     * print the results
     */
    printf("The string \"");
    while (idx < MAX_STRING_LENGTH && string[idx] != '\n' && string[idx] != '\0'){
        printf("%c", string[idx]);
        ++idx;
    }
    printf("\" is %sa palindrome !!", is_palindrome ? "" : "NOT ");

    return EXIT_CODE_OK;
}

unsigned int get_length(char * string){
    /**
     * Traverse the string as long as we dont overflow,
     * nor encounter new line or null
     */
    unsigned int idx = 0;
    while (idx < MAX_STRING_LENGTH && string[idx] != '\n' &&
           string[idx] != '\0'){
        ++idx;
    }
    return idx;
}

int is_whitespace(char character){
    /**
     * True if the character is a whitespace
     */
    return character == '\t' ||
           character == ' ';
}

int palindrome(char s[]){
    /**
     * Check if s is a palindrome
     */

    /* True by default */
    int is_palindrome = 1;

    unsigned int string_length; /* The length of s */
    int left_cursor;            /* a cursor into s going left to right */
    int right_cursor;           /* a cursor into s going right to left */
    char left_char;             /* the character the left cursor is pointing at */
    char right_char;            /* the character the right cursor is pointing at */

    /* Initialize the cursors according to the length of s */
    string_length = get_length(s);
    left_cursor = 0;
    right_cursor = string_length - 1;

    /**
     * As long as the string is still considered a palindrom,
     * And as long as the cursors are not exhausted,
     * And when the characters that correspond to the cursors are not whitespaces
     * Check the characters for equality
     *   - if they are not equal, it means that the string is not a palindrome
     */
    while ((is_palindrome == 1) && (left_cursor < right_cursor)){
        left_char = s[left_cursor];
        right_char = s[right_cursor];

        if (is_whitespace(left_char)){
            ++left_cursor;
        } else if (is_whitespace(right_char)){
            --right_cursor;
        } else {
            is_palindrome = (left_char == right_char);
            ++left_cursor;
            --right_cursor;
        }
    }

    return is_palindrome;
}