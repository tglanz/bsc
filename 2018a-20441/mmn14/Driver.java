public class Driver
{
    public static void main(String[] args)
    {
        // int counted = Ex14.count(new int[] { 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3 }, 2);
        // System.out.println("Counted: " + counted);

        /*
        String s;
        s = "00011011";
        System.out.println("Alts of " + s + " should be 2: " + Ex14.alternating(s));
        s = "0101";
        System.out.println("Alts of " + s + " should be 0: " + Ex14.alternating(s));
        s = "1100";
        System.out.println("Alts of " + s + " should be 1: " + Ex14.alternating(s));
        s = "00101011";
        System.out.println("Alts of " + s + " should be 1: " + Ex14.alternating(s));
        */

        /*
        System.out.println("A isWay should be true: " + Ex14.isWay(new int[] { 2, 4, 1, 6, 4, 2, 4, 3, 5 }));
        System.out.println("B isWay should be false: " + Ex14.isWay(new int[] { 1, 4, 3, 1, 2, 4, 3 }));
        System.out.println("C isWay should be false: " + Ex14.isWay(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }));
        */

        int[][] mat = new int[][] {
            new int[] { 3, 8, 7, 1 },
            new int[] { 5, 15, 2, 40 },
            new int[] { 12, 11, 13, 22 },
            new int[] { 3, 14, 16, 17, 62 }
        };

        Ex14.printPath(mat);
    }
}