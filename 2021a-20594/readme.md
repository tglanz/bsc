## notes

- try to avoid emulation; use gcc with 32 bit flag

    gcc -m32 ...

- don't use gcc version above 6.3

- abort using ```perror```

- check return values of system calls

- free all resources

- for make, supply a ```makefile```. not others

- use -Wall when compiling



When making xv6 we can get permissions denied on perl files. this can be resolved just by adding executions permissions.

    find . -name "*.pl" | xargs chmod +x
