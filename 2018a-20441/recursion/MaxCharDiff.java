public class MaxCharDiff
{
    public static void main(String[] args)
    {
        String s1, s2;
        int max;        

        s1 = "abc";
        s2 = "abc";
        max = 1;
        System.out.println(s1 + " - " + s2 + " - max: " + max + " IS: " + maxCharDiff(s1, s2, max));

        s1 = "abc";
        s2 = "bbc";
        max = 1;
        System.out.println(s1 + " - " + s2 + " - max: " + max + " IS: " + maxCharDiff(s1, s2, max));

        s1 = "abc";
        s2 = "abd";
        max = 1;
        System.out.println(s1 + " - " + s2 + " - max: " + max + " IS: " + maxCharDiff(s1, s2, max));

        s1 = "abc";
        s2 = "bcc";
        max = 1;
        System.out.println(s1 + " - " + s2 + " - max: " + max + " IS: " + maxCharDiff(s1, s2, max));

        s1 = "abc";
        s2 = "bcc";
        max = 2;
        System.out.println(s1 + " - " + s2 + " - max: " + max + " IS: " + maxCharDiff(s1, s2, max));

        s1 = "abc";
        s2 = "xyz";
        max = 3;
        System.out.println(s1 + " - " + s2 + " - max: " + max + " IS: " + maxCharDiff(s1, s2, max));
    }

    private static boolean maxCharDiff(String s1, String s2, int max)
    {
        // Assume s1.len == s2.len
        if (s1.length() == 0)
        {
            return true;
        }

        if (max < 0)
        {
            return false;
        }

        int nextMax = max;
        if (s1.charAt(0) != s2.charAt(0))
        {
            nextMax--;
        }

        return maxCharDiff(s1.substring(1), s2.substring(1), nextMax);
    }
}