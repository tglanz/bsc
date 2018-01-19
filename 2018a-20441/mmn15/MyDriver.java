public class MyDriver {
    public static void main(String[] args){
        String baseA = "aaaaabbbbaabababababa";
        String baseB = "0333339";
        StringList a, b;

        a = new StringList(baseA);
        b = new StringList(baseB);


        System.out.println(a.length() + " - " + baseA.length());
    }
}