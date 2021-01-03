package bloomfilter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.LinkedList;
import java.util.logging.Formatter;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Program {

    public static void main(String[] args) {
        Arguments arguments = new Arguments(args);
        validateArguments(arguments);

        BloomFilter<String> bloomFilter = createBloomFilter(arguments.bitCount, arguments.hashCount);

        try {
            // in case the numCount is specified, we will just add all
            // integers from 0 to numCount
            if (arguments.numCount > 0) {
                IntStream.range(0, arguments.numCount).boxed()
                    .map(Object::toString).forEach(bloomFilter::add);
            } else {
                // otherwise, work with the add and check input files
                readInput(arguments.addPath).forEach(bloomFilter::add);
                readInput(arguments.checkPath).forEach(item -> System.out.println(
                    String.format("contains(%s) = %s", item, bloomFilter.contains(item))));
            }

            System.out.println(String.format(
                        "current bloom filter's error: %.9f", bloomFilter.getError()));
        } catch (Exception exception) {
            System.err.println(exception.getMessage());
            System.exit(1);
        }
    }

    private static void validateArguments(Arguments arguments) {
        if (arguments.numCount > 0) {
            return;
        }

        if (arguments.addPath == null) {
            System.err.println("argument not specified: add-path");
            System.exit(1);
        }

        if (arguments.checkPath == null) {
            System.err.println("argument not specified: check-path");
            System.exit(1);
        }

        for (Path path : new Path[] { arguments.addPath, arguments.checkPath }) {
            if (!path.toFile().exists()) {
                System.err.println("no such file: " + path);
                System.exit(1);
            }
        }
    }

    private static BloomFilter<String> createBloomFilter(int bitCount, int hashCount) {
        Collection<BloomFilter.Default.HashFunction> hashes = new LinkedList<>();
        for (int idx = 0; idx < hashCount; ++idx) {
            hashes.add(new MurmurHash(idx)::hashString);
        }

        return new BloomFilter.Default(bitCount, hashes);
    }

    private static Stream<String> readInput(Path path)
            throws IOException {
        return Files.lines(path)
            .flatMap(line -> Stream.of(line.split(",", 0)))
            .map(String::trim);
    }
}
