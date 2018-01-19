public class MyDriver {
    public static void main(String[] args){
        StringList a, b;

        a = new StringList("01234");
        for (int idx = 0; idx < 6; ++idx){
            System.out.println("Char at " + idx + ": " + a.charAt(-1));
        }

    }
}