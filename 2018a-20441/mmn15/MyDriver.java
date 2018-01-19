public class MyDriver {
    public static void main(String[] args){
        String baseA = "1331339";
        String baseB = "0333339";
        StringList a, b;

        a = new StringList(baseA);
        b = new StringList(baseB);

        int compared = a.compareTo(b);
        String bbb = "Equal";

        if (compared < 0){
            bbb = "a < b";
        } else if (compared > 0){
            bbb = "a > b";
        }

        System.out.println("Compared: " + bbb);
    }
}