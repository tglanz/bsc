public class Q1 
{
    public int myFunc(char x, int y)
    {
        // this is the function to be overloaded
        return 0;
    }

    /**
     * The fuctions below are the ones to be checked.
     */

    public int myFunc()
    {
        return 0;
    }

    public double myFunc(char y, int x)
    {
        // This is the only method that should break compilation
        return 0;
    }

    public void myFuc(int x)
    {
    }

    public double myFunc(double a, double b)
    {
        return 0;
    }
}