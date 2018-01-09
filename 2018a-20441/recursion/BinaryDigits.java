public class BinaryDigits
{
    public static void main(String[] args)
    {
        binaryDigits(3);
    }

    private static void binaryDigits(int n)
    {
        binaryDigits(n, "");
    }

    private static void binaryDigits(int n, String str)
    {
        if (n == 0)
        {
            System.out.println(str);
        }
        else
        {
            binaryDigits(n - 1, "0" + str);
            binaryDigits(n - 1, "1" + str);
        }
    }
}