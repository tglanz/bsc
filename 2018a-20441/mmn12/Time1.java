/**
 * Represents time - hours:minutes. Coordinates cannot be negative.
 */
public class Time1
{
    private static final int MAX_HOUR = 23;
    private static final int MIN_HOUR = 0;
    private static final int DEFAULT_HOUR = 0;

    private static final int MAX_MINUTE = 59;
    private static final int MIN_MINUTE = 0;
    private static final int DEFAULT_MINUTE = 0;

    private static final int MINUTES_IN_HOUR = 60;

    private int _hour;
    private int _minute;    

    /**
     * Constructs a Time1 object.
     */
    public Time1(int h, int m)
    {
        // Default initialization
        _hour = DEFAULT_HOUR;
        _minute = DEFAULT_MINUTE;

        // Check the validity of h and m as hour and minute representatives
        if (isValidHour(h))
        {
            _hour = h;
        }

        if (isValidMinute(m))
        {
            _minute = m;
        }
    }

    /**
     * Copy constructor for Time1.
     */
    public Time1(Time1 t)
    {
        // Copy the fields themselves
        _hour = t._hour;
        _minute = t._minute;
    }

    /**
     * Returns the hour of the time.
     * @return int The hour of the time
     */
    public int getHour()
    {
        return _hour;
    }

    /**
     * Returns the minute of the time.
     * @return int The minute of the time
     */
    public int getMinute()
    {
        return _minute;
    }

    /**
     * Changes the hour of the time. If an illegal number is received hour will be unchanged.
     * @param num The new hour
     */
    public void setHour(int num)
    {
        if (isValidHour(num))
        {
            _hour = num;
        }
        
        // else remain unchanged
    }

    /**
     * Changes the minute of the time. If an illegal number is received minute will be unchanged.
     * @param num The new minute
     */
    public void setMinute(int num)
    {
        if (isValidMinute(num))
        {
            _minute = num;
        }
        
        // else remain unchanged
    }

    /**
     * Return a string representation of this time (hh:mm).
     * @return String representation of this time (hh:mm).
     */
    public String toString()
    {
        String hourString = "";
        String minuteString = "";

        int hour = getHour();
        int minute = getMinute();

        if (hour < 10)
        {
            hourString += "0";
        }

        if (minute < 10)
        {
            minuteString += "0";
        }

        hourString += hour;
        minuteString += minute;

        return hourString + ":" + minuteString;
    }

    /**
     * Return the amount of minutes since midnight.
     * @return int amount of minutes since midnight
     */
    public int minFromMidnight()
    {
        return getHour() * MINUTES_IN_HOUR + getMinute();
    }

    /**
     * Check if the received time is equal to this time.
     * @param other The time to be compared with this time
     * @return boolean True if the received time is equal to this time
     */
    public boolean equals(Time1 other)
    {
        return minFromMidnight() == other.minFromMidnight();
    }

    /**
     * Check if this time is before a received time.
     * @param other The time to check if this time is before
     * @return boolean True if this time is before other time
    */
    public boolean before(Time1 other)
    {
        return minFromMidnight() < other.minFromMidnight();
    }

    /**
     * Check if this time is after a received time.
     * @param other The time to check if this time is before
     * @return boolean True if this time is after other time
    */
    public boolean after(Time1 other)
    {
        return other.before(this);
    }

    public int difference(Time1 other)
    {
        // return (getHour() - other.getHour()) * MINUTES_IN_HOUR + (getMinute() - other.getMinute());
        return minFromMidnight() - other.minFromMidnight();
    }

    /* Private methods */

    private static boolean isInRange(int value, int min, int max)
    {
        return min <= value && value <= max;
    }

    private static boolean isValidHour(int h)
    {
        return isInRange(h, MIN_HOUR, MAX_HOUR);
    }

    private static boolean isValidMinute(int m)
    {
        return isInRange(m, MIN_MINUTE, MAX_MINUTE);
    }
}
