# mmn 14

how to build

    javac -d target $(find src/ -name "*.java")    

how to run

    java -cp target bloomfilter.Program [OPTIONS]

example - show help

    java -cp target bloomfilter.Program --help

example - run with K=13, m=32000000 and N=1000000 arbitrary numbers

    java -cp target bloomfilter.Program -k 13 -m 32000000 -n 1000000

example - run with K=13, m=32000000, add words based on data/input.txt and check words based on data/check.txt

    java -cp target bloomfilter.Program -k 13 -m 32000000 -a data/input.txt -c data/check.txt
