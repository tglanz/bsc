public class MyDriver {
    public static void main(String[] args){
        String baseA = "835";
        StringList a, b;

        a = new StringList(baseA);

        System.out.println(a.indexOf('0'));
        System.out.println(a.indexOf('1'));
        System.out.println(a.indexOf('2'));
        System.out.println(a.indexOf('3'));
        System.out.println(a.indexOf('4'));
        System.out.println(a.indexOf('5'));
        System.out.println(a.indexOf('6'));
        System.out.println(a.indexOf('7'));
        System.out.println(a.indexOf('8'));
        System.out.println(a.indexOf('9'));
    }
}