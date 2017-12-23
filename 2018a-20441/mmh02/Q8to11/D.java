public class D extends C
{
    public void foo(C c)
    {
        System.out.println("dc");
    }
    public void foo(D d)
    {
        System.out.println("dd");
    }
}
