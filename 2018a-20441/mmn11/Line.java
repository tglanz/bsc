import java.util.Scanner;

/*
    Class    : Line.java
    Author   : Tal Glanzman
    Summuary : Calculate the distance between two 2d points
*/
public class Line
{
    public static void main(String[] args)
    {
        // Acquire input from the user
        Scanner scan = new Scanner(System.in);
        System.out.println("Please enter 4 integers");
        
        System.out.println("Please enter x1:");
        int x1 = scan.nextInt();
        System.out.println("Please enter y1:");
        int y1 = scan.nextInt();
        System.out.println("Please enter x2:");
        int x2 = scan.nextInt();
        System.out.println("Please enter x2:");
        int y2 = scan.nextInt();
        
        // Calculate the distance using the formula: [(x1-x2)^2 + (y1-y2)^2]^.5
        double xDeltaSquared = Math.pow(x1 - x2, 2);
        double yDeltaSquared = Math.pow(y1 - y2, 2);
        double distance = Math.sqrt(xDeltaSquared + yDeltaSquared);
        
        // Output the result to the user
        System.out.println(
            "The length of the line between the points " +
            "(" + x1 + "," + y1 + ") and " +
            "(" + x2 + "," + y2 + ") is " +
            distance
        );
    } // end of method main
} // end of class Line
