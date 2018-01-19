public class MyDriver {
    public static void main(String[] args){
        String baseA = "835";
        StringList a, b;

        a = new StringList(baseA);

        for (char c = '0'; c <= '9'; c++){
            System.out.println("character " + c + " is at index " + a.indexOf(c, 2));
        }
    }
}