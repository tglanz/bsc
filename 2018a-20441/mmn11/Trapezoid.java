import java.util.Scanner;

public class Trapezoid
{
    public static void main(String[] args)
    {
        // Acquire the input from the user
        Scanner scanner = new Scanner(System.in);
        System.out.println("Please enter 6 integers");

        // The inputs for the first base
        System.out.print("First base left point x coordinate: ");
        int firstBaseLeftX = scanner.nextInt();
        System.out.print("First base y coordinate: ");
        int firstBaseY = scanner.nextInt();
        System.out.print("First base width: ");
        int firstBaseWidth = scanner.nextInt();

        // The inputs for the second base
        System.out.print("Second base left point x coordinate: ");
        int secondBaseLeftX = scanner.nextInt();
        System.out.print("Second base y coordinate: ");
        int secondBaseY = scanner.nextInt();
        System.out.print("Second base width: ");
        int secondBaseWidth = scanner.nextInt();

        // Calculate the right point coordinates for further use
        double firstBaseRightX = firstBaseLeftX + firstBaseWidth;
        double secondBaseRightX = secondBaseLeftX + secondBaseWidth;

        /*
            Calculate the perimeter.
            perimeter = 
                sum of the width of the bases +
                length of the sides of the trapezoid

            We will calculate the length of the sides using the distance between two point formula
            which was introduced in Line.java excercise
        */
        double leftSideLength = Math.sqrt(
            Math.pow(firstBaseLeftX - secondBaseLeftX, 2) + 
            Math.pow(firstBaseY - secondBaseY, 2));

        double rightSideLength = Math.sqrt(
            Math.pow(firstBaseRightX - secondBaseRightX, 2) + 
            Math.pow(firstBaseY - secondBaseY, 2));

        double perimeter = firstBaseWidth + secondBaseWidth + leftSideLength + rightSideLength;

        /*
            Calculate the area.
            area =
                sum of the width of the bases *
                height of the trapezoid *
                .5

            In order to calculate the height WITHOUT using Math.abs we will again use the distance formula
            on the points (0, y1) and (0, y2) where y1 and y2 are the y coordinates of the bases.
            So the formula is
                [(0-0)^2 + (y1 - y2)^2]^.5 == (y1 - y2)^2^.5
        */
        double height = Math.sqrt(Math.pow(firstBaseY - secondBaseY, 2));
        double area = (firstBaseWidth + secondBaseWidth) * height * .5;

        // Output the results to the user
        System.out.println("The area of the trapezoid is " + area);
        System.out.println("The perimeter of the trapezoid is " + perimeter);
    }
}