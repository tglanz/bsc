public class Student extends Person
{
    private int[] _grades;
    private int _id;
    public static int _studentsNum;
    public Student (Student student)
    {
        super(student._name);
    }
}