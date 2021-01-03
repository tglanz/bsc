# mmn 14

how to build

    javac -d target $(find src/ -name "*.java")    

how to run

    java -cp target bloomfilter.Program [OPTIONS]

below is the output of ```java -cp target bloomfilter.Program --help```

```bash
usage: java bloomfilter.Program [OPTIONS]

OPTIONS
  -h,--help            show this message

  -k,--hash-count      determines the number of hash functions to use
                       default: 13
  -m,--bit-count       determines the number of bits to use
                       default: 32000000

  -n,--number-count    mostly for debug purposes, if specified,
                       add to the bloom filter integers from 0 to number-count
                       and disable all file handling.

  -a,--add-path        path to a file containing items to add
  -c,--check-path      path to a file containing items to check

EXAMPLES
  # using 13 hash functions and a 32Mbit container, add 1 mil numbers to the bloom
  # practically, use the parameters like in the exercice - used to validate error
  java bloomfilter.Program --hash-count 13 --bit-count 32000000 --number-count 1000000

  # using 13 hash functions and a 32Mbit container, add all values in data/check.txt and check
  # for containment of all values in data/check.txt
  java bloomfilter.Program -k 13 -m 32000000 -a data/input.txt -c data/check.txt 
```
