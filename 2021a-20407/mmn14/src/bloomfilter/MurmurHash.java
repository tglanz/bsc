package bloomfilter;

public class MurmurHash {

    private final int seed;

    public MurmurHash(int seed) {
        this.seed = seed;
    }

    /**
     * compute the has of a string.
     * 
     * complexity: O(N)
     *     where N is the length of the string
     */
    public int hashString(String key) {
        return hash(key.getBytes(), seed);
    }

    /**
     * this was taken from
     * https://github.com/Litarvan/JavaMurmur/blob/master/src/main/java/fr/litarvan/javamurmur/MurmurHasher.java
     */
    public static int hash(byte[] key, int seed)
    {
        int klen = key.length;
        int remainder;
        int i, bytes;

        int h1 = seed;
        for (i = 0, bytes = klen - (remainder = klen & 3); i < bytes;)
        {
            int k1 = (((int) key[i] & 0xff)) | (((int) key[++i] & 0xff) << 8) | (((int) key[++i] & 0xff) << 16) | (((int) key[++i] & 0xff) << 24);
            ++i;
            k1 = ((((k1 & 0xffff) * 0xcc9e2d51) + (((((k1 >= 0 ? k1 >> 16 : ((k1 & 0x7fffffff) >> 16) | 0x8000)) * 0xcc9e2d51) & 0xffff) << 16)));
            k1 = k1 << 15 | (k1 >= 0 ? k1 >> 17 : ((k1 & 0x7fffffff) >> 17) | 0x4000);
            k1 = ((((k1 & 0xffff) * 0x1b873593) + (((((k1 >= 0 ? k1 >> 16 : ((k1 & 0x7fffffff) >> 16) | 0x8000)) * 0x1b873593) & 0xffff) << 16)));
            h1 ^= k1;
            h1 = h1 << 13 | (h1 >= 0 ? h1 >> 19 : ((h1 & 0x7fffffff) >> 19) | 0x1000);
            int h1b = ((((h1 & 0xffff) * 5) + (((((h1 >= 0 ? h1 >> 16 : ((h1 & 0x7fffffff) >> 16) | 0x8000)) * 5) & 0xffff) << 16)));
            h1 = (((h1b & 0xffff) + 0x6b64) + (((((h1b >= 0 ? h1b >> 16 : ((h1b & 0x7fffffff) >> 16) | 0x8000)) + 0xe654) & 0xffff) << 16));
        }
        int k1 = 0;
        switch (remainder)
        {
            case 3:
                k1 ^= ((int) key[i + 2] & 0xff) << 16;
            case 2:
                k1 ^= ((int) key[i + 1] & 0xff) << 8;
            case 1:
                k1 ^= ((int) key[i] & 0xff);
                k1 = (((k1 & 0xffff) * 0xcc9e2d51) + (((((k1 >= 0 ? k1 >> 16 : ((k1 & 0x7fffffff) >> 16) | 0x8000)) * 0xcc9e2d51) & 0xffff) << 16));
                k1 = k1 << 15 | (k1 >= 0 ? k1 >> 17 : ((k1 & 0x7fffffff) >> 17) | 0x4000);
                k1 = (((k1 & 0xffff) * 0x1b873593) + (((((k1 >= 0 ? k1 >> 16 : ((k1 & 0x7fffffff) >> 16) | 0x8000)) * 0x1b873593) & 0xffff) << 16));
                h1 ^= k1;
        }
        h1 ^= klen;
        h1 ^= (h1 >= 0 ? h1 >> 16 : ((h1 & 0x7fffffff) >> 16) | 0x8000);
        h1 = (((h1 & 0xffff) * 0x85ebca6b) + (((((h1 >= 0 ? h1 >> 16 : ((h1 & 0x7fffffff) >> 16) | 0x8000)) * 0x85ebca6b) & 0xffff) << 16));
        h1 ^= (h1 >= 0 ? h1 >> 13 : ((h1 & 0x7fffffff) >> 13) | 0x40000);
        h1 = ((((h1 & 0xffff) * 0xc2b2ae35) + (((((h1 >= 0 ? h1 >> 16 : ((h1 & 0x7fffffff) >> 16) | 0x8000)) * 0xc2b2ae35) & 0xffff) << 16)));
        h1 ^= (h1 >= 0 ? h1 >> 16 : ((h1 & 0x7fffffff) >> 16) | 0x8000);

        return h1;
    }
}
