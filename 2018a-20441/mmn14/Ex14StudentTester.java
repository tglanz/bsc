
/**
 * Tester for MMN 14
 * Please make sure your code runs with this tester *WITHOUT* any errors
 * 
 * Place this file in the same directory as Ex14.java code you wrote
 * 
 * Good luck!
 */
public class Ex14StudentTester
{
    /**
     * Main method
     */
    public static void main(String[] args)
    {
        // Testing Q1 - count
        System.out.println("Testing Question 1\n==================");
        int[] a = {-5,-5,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,67,67,99};
        System.out.println("a = {-5,-5,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,67,67,99}");
        System.out.println("Ex14.count(a,2) is: " + Ex14.count(a,2));
        
        // Testing Q2 - alternating
        System.out.println("\nTesting Question 2\n==================");
        System.out.println("Ex14.alternating(\"00011011\") is: " + Ex14.alternating("00011011"));
        
        // Testing Q3 - isWay
        System.out.println("\nTesting Question 3\n==================");
        int[] a1 = {2,4,1,6,4,2,4,3,5};
        System.out.println("a = {2,4,1,6,4,2,4,3,5}");
        System.out.println("Ex14.isWay(a) is: " + Ex14.isWay(a1));
        
        // Testing Q4 - printPath
        System.out.println("\nTesting Question 4\n==================");
        int [][] mat = {{3,8,7,1},{5,15,2,4},{2,1,3,2},{4,6,7,52}};
        System.out.println("mat={\t{3,8,7,1}\n\t{5,15,2,4}\n\t{2,1,3,2}\n\t{4,6,7,52}\n    }");
        System.out.println("Ex14.printPath(mat) output is:");
        Ex14.printPath(mat);
    }
}
