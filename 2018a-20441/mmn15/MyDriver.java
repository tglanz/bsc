public class MyDriver {
    public static void main(String[] args){
        StringList a, b;

        a = new StringList("01234");
        b = new StringList("abcde");
        
        System.out.println("COncat");
        a.concat(b).DEBUG_PrintStuff();

        System.out.println("A");
        a.DEBUG_PrintStuff();

        System.out.println("B");
        b.DEBUG_PrintStuff();
    }
}