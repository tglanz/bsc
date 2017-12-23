import javax.rmi.CORBA.ClassDesc;

/**
 * Let A be an abstract class and B extends A is some class.
 * Here we will try to prove that the only TRUE statement below is 4.
 * 1. Every method in A should abstract
 * 2. A doesn't have a constructor
 * 3. B, in any case, must implement every abstract method of A
 * 4. It could be, that in A, there is no abstract method
 */

public class Q5
{
    public class ContradictSentence1
    {
        public abstract class A
        {
            public abstract void abstractMethod();

            public void nonAbstractMethod()
            {
            }
        }
    }

    public class ContradictSentence2
    {
        public abstract class A
        {
            public A()
            {
            }
        }
    }

    public class ContradictSentence3
    {
        public abstract class A
        {
            public abstract void someFunc();
        }

        public abstract class B
        {
            // This is an example of a class that doesn't need to implement nothing
            // It's becuase its abstract itself
        }
    }

    public class ProveSentence4
    {
        public abstract class A
        {
            public void nonAbstractMethod()
            {

            }
        }
    }
}