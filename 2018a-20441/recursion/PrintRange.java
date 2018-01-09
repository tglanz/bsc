public class PrintRange
{
    public static void main(String[] args)
    {
        printRange2(10);
    }

    public static void printRange(int count)
    {
        printRange(1, count);
    }

    public static void printRange(int current, int count)
    {
        if (count > 0)
        {
            System.out.println(current);
            printRange(current + 1, count - 1);
        }
    }

    public static void printRange2(int count)
    {
        if (count > 0)
        {
            printRange2(count - 1);
            System.out.println(count);
        }
    }
}