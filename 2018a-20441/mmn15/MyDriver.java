public class MyDriver {
    public static void main(String[] args){
        String baseA = "";
        String baseB = "";
        StringList a, b;

        a = new StringList(baseA);
        b = new StringList(baseB);

        System.out.println("Equals: " + a.equals(b));
    }
}