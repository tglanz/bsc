public class Driver
{
    public static void main (String [] args)
    {
        C cd = new D();
        D dd = (D)cd;

        System.out.print("8. Expecting dd: ");
        dd.foo(dd);

        System.out.print("9. Expecting dc: ");
        dd.foo(cd);

        System.out.print("10. Expecting dd: ");
        cd.foo(dd);

        // COMPILATION ERROR EXPECTED
        // cd.foo(cd);
    }
}
