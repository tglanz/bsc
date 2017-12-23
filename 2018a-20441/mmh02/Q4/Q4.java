public class Q4
{
    public static void main(String[] args)
    {
        Classes classes = new Classes();
        Classes.C c = classes.new C();

        System.out.println(c.getX());
    }
}