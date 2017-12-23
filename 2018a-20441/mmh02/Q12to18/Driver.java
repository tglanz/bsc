public class Driver {
    public static void main(String[] args) {
        AA a1 = new AA();
        AA a2 = new BB();
        AA a3 = new AA();
        AA a4 = new BB();
        BB b1 = new BB();
        BB b2 = new BB();

        System.out.println("12. expecting x, false: "); 
        System.out.println(a3.equals(a1));

        System.out.println("13. expecting 1, true: ");
        System.out.println(a4.equals(a2));

        System.out.println("14. expecting x, false: ");
        System.out.println(a1.equals(a2));

        System.out.println("15. expecting 1, true: ");
        System.out.println(a2.equals(b1));

        System.out.println("16. expecting 2, false: ");
        System.out.println(b1.equals(a1));

        System.out.println("17. expecting : 3, true");
        System.out.println(b2.equals(b1));

        System.out.println("18. expecting : 2, true");
        System.out.println(b1.equals(a4));
    }
}
