public class TimesPrint
{
    public static void main(String[] args)
    {
        timesPrint('*', 10);
    }

    private static void timesPrint(char c, int n)
    {
        if (n > 0)
        {
            System.out.print(c);
            timesPrint(c, n - 1);
        }
    }
}