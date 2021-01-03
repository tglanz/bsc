package bloomfilter;

import java.util.BitSet;
import java.util.Collection;
import java.util.function.Function;

/**
 * BloomFilter API.
 * 
 * list the functions that each implementation must follow, along the complexity gaurantees.
 */
public interface BloomFilter<T> {

    /**
     * adds an item to the construct.
     * 
     * complexity: O(K * P) where
     *      K is the number of hash functions the construct uses
     *      P is the time complexity of a hash function
     * 
     * 
     * @param item item to add
     */
    void add(T item);

    /**
     * checks whether some item is contained in the construct.
     * 
     * complexity: O(K * P) where
     *      K is the number of hash functions the construct uses
     *      P is the time complexity of a hash function
     * 
     * @param item item to check if is contained
     * @return
     *      if false, false is not contained.
     *      if true, the item can be or not be contained in the construct;
     *      the chance for false positives can be given by {@link BloomFilter#error()}
     */
    boolean contains(T item);

    /**
     * calculates the chance for false positives.
     * 
     * complexity: O(1)
     * 
     * i.e; determines what is the chance for a non-existing item is thought of as exiting.
     * 
     * @return probability of false positives
     */
    double getError();

    /**
     * a standard implementation of {@link BloomFilter}.
     * 
     * this implementation is invariant to the hash implementation which is parametrized.
     * 
     * the gaurantees of runtime complexities are described in the interface definition {@link BloomFilter}
     */
    public static class Default implements BloomFilter<String> {

        /**
         * the hash functions prototype this implementation requires.
         * 
         * practically, just a function from integer to integer
         */
        public interface HashFunction extends Function<String, Integer> { }

        /** the hash functions to apply in order */
        private final Collection<HashFunction> hashes;

        /** the data container */
        private final BitSet bitset;

        // variables to compute the error according to.
        // we take advantage of the fact that the probability can prtially be computed incrementally,
        // while the rest of the computation can be done at O(K) time.
        // the computations will be added to the add function, which is O(K) time already, so complexity impact.
        private final double innerErrorCoefficient;      // ((m-1)/m)^K
        private double innerError;                       // ((m-1)/m)^KN)
        private double error;                            // (1 - ((m-1)/m)^(NK))^K

        public Default(int bitCount, Collection<HashFunction> hashes) {
            this.hashes = hashes;
            this.bitset = new BitSet(bitCount);
            this.innerErrorCoefficient = Math.pow((bitCount - 1) / (double)bitCount, hashes.size());
            this.innerError = 1;
            this.error = 0;
        }

        public void add(String item) {
            if (item == null) {
                throw new IllegalArgumentException("adding null is prohibited");
            }

            // used to recompute the error
            innerError *= innerErrorCoefficient;
            error = 1;

            // this loop is the reason for O(K) time
            for (HashFunction hash : hashes) {
                // compute the hash
                int bit = Math.abs(hash.apply(item)) % bitset.size();

                // set the bit
                bitset.set(bit);

                // incrementally compute the error
                error *= (1D - innerError);
            }
        }

        public boolean contains(String item) {
            if (item == null) {
                throw new IllegalArgumentException("null items are prohibited");
            }

            // this loop is the reason for O(K) time
            for (HashFunction hash : hashes) {
                // compute the hash
                int bit = Math.abs(hash.apply(item)) % bitset.size();

                // we can short circuit if there is at least one bit off.
                // doesn't change the worst case complexity, but is practically better.
                if (!bitset.get(bit)) {
                    return false;
                }
            }

            return true;
        }

        public double getError() {
            // just return a member; O(1) ofcourse
            return error;
        }
    }
}
