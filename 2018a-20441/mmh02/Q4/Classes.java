public class Classes
{
    public class A
    {
        protected int _x;

        protected A()
        {
            _x = 123;
        }
    }

    public class B extends A
    {

    }

    public class C extends B
    {
        public int getX()
        {
            return _x;
        }
    }
}