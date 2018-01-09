public class OneTwoSums
{
    public static void main(String[] args)
    {
        int num = 0;
        System.out.println("Options of " + num);
        printOptions(num);
    }

    public static void printOptions(int num)
    {
        printOptions(num, 0, "");
    }

    public static void printOptions(int num, int a, String ln)
    {
        if (a > num)
        {
            return;
        }

        if (a == num)
        {
            System.out.println(ln);
        }
        else
        {
            printOptions(num, a + 1, ln + "1 ");
            printOptions(num, a + 2, ln + "2 ");
        }
    }
}