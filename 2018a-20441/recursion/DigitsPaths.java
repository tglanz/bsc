public class DigitsPaths
{
    public static void main(String[] args)
    {
        int[] a = new int[] { 34, 59, 74, 12, 15, 17 };

        System.out.println("Path exists : " + pathExist(a));
        System.out.println("How many    : " + howManyPathsExist(a));
    }

    private static boolean pathExist(int[] a)
    {
        return pathExist(a, 0);
    }

    private static boolean pathExist(int[] a, int pos)
    {
        if (pos == a.length - 1)
        {
            return true;
        }

        if (pos > a.length - 1)
        {
            return false;
        }

        int ones = a[pos] % 10;
        int tens = (a[pos] / 10) % 10;

        if (ones != 0 && pathExist(a, pos + ones))
        {
            return true;
        }

        return tens != 0 && pathExist(a, pos + tens);
    }

    private static int howManyPathsExist(int[] a)
    {
        return howManyPathsExist(a, 0);
    }

    private static int howManyPathsExist(int[] a, int pos)
    {
        int ones = a[pos] % 10;
        int tens = (a[pos] / 10) % 10;

        return howManyPathsExist(a, pos, ones) + howManyPathsExist(a, pos, tens);
    }

    private static int howManyPathsExist(int[] a, int pos, int step)
    {
        if (step == 0)
        {
            return 0;
        }

        int nextPos = pos + step;

        if (nextPos == a.length - 1)
        {
            return 1;
        }

        if (nextPos > a.length - 1)
        {
            return 0;
        }

        return howManyPathsExist(a, nextPos);
    }
}