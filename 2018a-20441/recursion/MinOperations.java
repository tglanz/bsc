public class MinOperations
{
    public static void main(String[] args)
    {
        int x, y;

        x = 10;
        y = 21;
        System.out.println("Min operations: (" + x + ", " + y + "): " + minOperations(x, y));
    }

    private static int minOperations(int x, int y)
    {
        if (x == y)
        {
            return 0;
        }

        if ((x * 2) <= y)
        {
            return 1 + minOperations(x * 2, y);
        }

        return 1 + minOperations(x + 1, y);
    }
}