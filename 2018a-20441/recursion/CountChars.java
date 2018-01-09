public class CountChars
{
    public static void main(String[] args)
    {
        String str = "aabaac";
        char c = 'a';

        System.out.println("There are " + countChars('a', str) + " ocurrences of 'a' in " + str);
        System.out.println("There are " + countChars('b', str) + " ocurrences of 'b' in " + str);
        System.out.println("There are " + countChars('c', str) + " ocurrences of 'c' in " + str);
        System.out.println("There are " + countChars('d', str) + " ocurrences of 'd' in " + str);
    }

    private static int countChars(char c, String str)
    {
        return countChars(c, str, 0);
    }

    private static int countChars(char c, String str, int location)
    {
        if (location >= str.length())
        {
            return 0;
        }

        int increment = 0;

        if (str.charAt(location) == c)
        {
            increment = 1;
        } 

        return increment + countChars(c, str, location + 1);
    }
}