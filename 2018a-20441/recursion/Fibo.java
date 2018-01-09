public class Fibo
{
    public static void main(String[] args)
    {
        fibo(5);
    }

    private static int fibo(int n)
    {
        if (n <= 1)
        {
            return 1;
        }

        int value = fibo(n - 1) + fibo(n - 2);

        return value;
    }
}