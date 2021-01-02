package bloomfilter;

import java.io.File;
import java.nio.file.Path;

class Arguments {

    public static int DEFAULT_HASH_COUNT = 13;
    public static int DEFAULT_BIT_COUNT = 32000000;

    public int hashCount;
    public int bitCount;
    public int numCount;
    public Path addPath;
    public Path checkPath;

    public Arguments(String[] args) {
        // set defaults
        hashCount = DEFAULT_HASH_COUNT;
        bitCount = DEFAULT_BIT_COUNT;
        numCount = 0;
        addPath = null;
        checkPath = null;

        for (int idx = 0; idx < args.length; ++idx) {
            String key = args[idx].toLowerCase();
            if (any(key, "-h", "--help")) {
                printUsage();
                System.exit(0);
            } else if (any(key, "-k", "--hash-count")) {
                idx++;
                hashCount = Integer.parseInt(args[idx]);
            } else if (any(key, "-m", "--bit-count")) {
                idx++;
                bitCount = Integer.parseInt(args[idx]);
            } else if (any(key, "-n", "--number-count")) {
                idx++;
                numCount = Integer.parseInt(args[idx]);
            } else if (any(key, "-a", "--add-path")) {
                idx++;
                addPath = Path.of(args[idx]);
            } else if (any(key, "-c", "--check-path")) {
                idx++;
                checkPath = Path.of(args[idx]);
            } else {
                throw new IllegalArgumentException("invalid program argument: " + key);
            }
        }
    }

    private static boolean any(String value, String... options) {
        for (String option : options) {
            if (value.equalsIgnoreCase(option)) {
                return true;
            }
        }

        return false;
    }

    public static void printUsage() {
        String program = "java bloomfilter.Program";
        
        String[] lines = new String[] {
            String.format("usage: %s [OPTIONS]", program),
            "",
            "OPTIONS",
            "  -h,--help            show this message",
            "",
            "  -k,--hash-count      determines the number of hash functions to use",
            "                       default: " + DEFAULT_HASH_COUNT,
            "  -m,--bit-count       determines the number of bits to use",
            "                       default: " + DEFAULT_BIT_COUNT,
            "",
            "  -n,--number-count    mostly for debug purposes, if specified,",
            "                       add to the bloom filter integers from 0 to number-count",
            "                       and disable all file handling.",
            "",
            "  -a,--add-path        path to a file containing items to add",
            "  -c,--check-path      path to a file containing items to check",
            "",
            "EXAMPLES",
            "  # using 13 hash functions and a 32Mbit container, add 1 mil numbers to the bloom",
            "  # practically, use the parameters like in the exercice - used to validate error",
            String.format("  %s --hash-count 13 --bit-count 32000000 --number-count 1000000", program),
            "",
            "  # using 13 hash functions and a 32Mbit container, add all values in data/check.txt and check",
            "  # for containment of all values in data/check.txt",
            String.format("  %s -k 13 -m 32000000 -a data/input.txt -c data/check.txt", program),
        };

        System.out.println(String.join("\n", lines));
    }
}
