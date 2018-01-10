public class Ex14
{
    /**
     * Counts the number of instances in an ordered array.
     * The method uses a binary search variation twice in order to look for the 
     * appearance of x in the ordered array.
     * 
     * Time complexity:     O(Log2 n)
     * Memory complexity:   O(1)
     * 
     * @param a ordered integers array to look into
     * @param x the value to search for
     * @return int number of instances of 'x' in 'a'
     */
    public static int count(int[] a, int x)
    {
        // [ 2, 2, 2, 3, 4, 4, 4, 5, 5, 6, 7, 8, 8, 9]

        // Check if the input is valid..
        if (a == null || a.length == 0)
        {
            return 0;
        }

        // If the min / max is bigger / lower than target value, it isn't in the array..
        if (a[0] > x || a[a.length - 1] < x)
        {
            return 0;
        }

        // Find the highest / lowest indices that x appears in the array
        int highIndex = countBinarySearchHigh(a, x);
        int lowIndex = countBinarySearchLow(a, x);

        return highIndex - lowIndex;
    }

    /**
     * Calculate the minimal number of swaps required to transform a
     * binary representation string into an alternating binary string
     * 
     * Time complexity      O(n)
     * Memory complexity    O(1)
     * 
     * @param s The string to get the number of swaps required to reach an alternating string
     * @return int The minimal number of swaps
     */
    public static int alternating(String s)
    {
        /* 
            The general algorithm and why it works:

            Calculate the number of differences of s to the alternating series 010101... (base).
            The number of differences of s to the alternating series 10101... is of course its complement.
            The number of swaps required to reach each alternating string is the half of the differences
            because each swap effectively fixes 2 characters positions.
            Return the minimum of those.
        */
        
        final String base = "01";
        int diffs = 0;

        // Compare each character in s to 0101...
        for (int i = 0; i < s.length(); i++)
        {
            if (s.charAt(i) != base.charAt(i % 2))
            {
                diffs++;
            }
        }

        // find the minimal differences, and divide it by 2 to get the swaps count
        return Math.min(diffs, s.length() - diffs) / 2;
    }

    /**
     * determines whether the is a path of steps from index 0 to last index,
     * based on the value in the indices
     * 
     * @param a the integers array to check if there are any paths
     * @param boolean true if there is a valid path, otherwise false
     */
    public static boolean isWay(int[] a)
    {
        return isWay(a, 0);
    }

    /**
     * recursion utility for isWay
     */
    private static boolean isWay(int[] a, int i)
    {
        // Sucess condition
        if (i == a.length - 1)
        {
            return true;
        }

        int current = a[i];

        // Check we are not overflowing
        boolean canGoRight = i + current < a.length;

        // Check we are not overflowing nor reaching endless recursion left
        boolean canGoLeft = i - current > 0 && a[i - current] != current;

        // recursion step
        return
            (canGoRight && isWay(a, i + a[i])) ||
            (canGoLeft && isWay(a, i - a[i]));
    }

    /**
     * Print a path from index [0,0] to a cell which contains a hill
     * 
     * @param mat the integers matrix to look for a path in
     */
    public static void printPath(int[][] mat)
    {
        String path = getPath(mat, 0, 0, "");
        System.out.println(path);
    }

    /**
     * printPath recursion utility
     */
    private static String getPath(int[][] mat, int row, int col, String path)
    {
        /*
            The algorithm is based on the fact that if there is a step up in
            a direction, we can follow it. Evantually we will get to a hill
            as long as we made valid steps.

            Basically the final path determined by the order of the valid movement, the "ifs" below.
        */

        path += "(" + row + "," + col + ")";

        // The value which we are at now
        int value = mat[row][col];

        // Can and should move up?
        if (row > 0 && mat[row - 1][col] > value)
        {
            return getPath(mat, row - 1, col, path);
        }

        // Can and should move right?
        if (col + 1 < mat[0].length && mat[row][col + 1] > value)
        {
            return getPath(mat, row, col + 1, path);
        }

        // Can and should move down?
        if (row + 1 < mat.length && mat[row + 1][col] > value)
        {
            return getPath(mat, row + 1, col, path);
        }

        // Can and should move left?
        if (col - 1 < 0 && mat[row][col - 1] > value)
        {
            return getPath(mat, row, col - 1, path);
        }

        // If none of the conditions above held; 
        // than we are currently at a hill and are done
        return path;
    }

    /**
     * Utility for count method.
     * Use a binary search variation to look for the highest index that a given value is in.
     */
    private static int countBinarySearchHigh(int[] a, int x)
    {
        int lo = 0;
        int hi = a.length - 1;
        int cursor = (hi - lo) / 2;

        while (cursor > lo)
        {
            if (a[cursor] > x)
            {
                hi = cursor;
            } 
            else
            {
                lo = cursor;
            }

            cursor = lo + ((hi - lo) / 2);
        }

        return cursor;
    }

    /**
     * Utility for count method.
     * Use a binary search variation to look for the lowest index that a given value is in.
     */
    private static int countBinarySearchLow(int[] a, int x)
    {
        int lo = 0;
        int hi = a.length - 1;
        int cursor = (hi - lo) / 2;

        while (cursor > lo)
        {
            if (a[cursor] >= x)
            {
                hi = cursor;
            } 
            else
            {
                lo = cursor;
            }

            cursor = lo + ((hi - lo) / 2);
        }

        return cursor;
    }
}