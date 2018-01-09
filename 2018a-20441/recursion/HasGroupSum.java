public class HasGroupSum
{
    public static void main(String[] args)
    {
        int sum = 17;
        int[] arr = new int[]{ 9, 2, 3, 7, 5 };
        
        System.out.println("Has group sum: " + hasGroupSum(arr, sum));
    }

    private static boolean hasGroupSum(int[] arr, int sum)
    {
        return hasGroupSum(arr, sum, 0);
    }

    private static boolean hasGroupSum(int[] arr, int sum, int pos)
    {
        if (sum == 0)
        {
            return true;
        }

        if (pos == arr.length)
        {
            return false;
        }

        return
            hasGroupSum(arr, sum - arr[pos], pos + 1) ||
            hasGroupSum(arr, sum, pos + 1);
    }
}