import javax.net.ssl.ExtendedSSLSession;

/**
 * Represents time - hours:minutes. Coordinates cannot be negative.
 */
public class Time2
{
    private static final int MAX_HOUR = 23;
    private static final int MIN_HOUR = 0;
    private static final int DEFAULT_HOUR = 0;

    private static final int MAX_MINUTE = 59;
    private static final int MIN_MINUTE = 0;
    private static final int DEFAULT_MINUTE = 0;

    private static final int MINUTES_IN_HOUR = 60;

    private int _minFromMid;

    /**
     * Constructs a Time2 object.
     */
    public Time2(int h, int m)
    {
        int hour = DEFAULT_HOUR;
        int minute = DEFAULT_MINUTE;

        if (isValidHour(h))
        {
            hour = h;
        }

        if (isValidMinute(m))
        {
            minute = m;
        }

        _minFromMid = calcMinFromMid(hour, minute);
    }

    /**
     * Copy constructor for Time2.
     */
    public Time2(Time2 t)
    {
        _minFromMid = t.minFromMidnight();
    }

    /**
     * Returns the hour of the time.
     * @return int The hour of the time
     */
    public int getHour()
    {
        return _minFromMid / MINUTES_IN_HOUR;
    }

    /**
     * Returns the minute of the time.
     * @return int The minute of the time
     */
    public int getMinute()
    {
        return _minFromMid % MINUTES_IN_HOUR;
    }

    /**
     * Changes the hour of the time. If an illegal number is received hour will be unchanged.
     * @param num The new hour
     */
    public void setHour(int num)
    {
        if (isValidHour(num))
        {
            _minFromMid = calcMinFromMid(num, getMinute());
        }
    }

    /**
     * Changes the minute of the time. If an illegal number is received minute will be unchanged.
     * @param num The new minute
     */
    public void setMinute(int num)
    {
        if (isValidMinute(num))
        {
            _minFromMid = calcMinFromMid(getHour(), num);
        }
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
        return _minFromMid;
    }

    /**
     * Check if the received time is equal to this time.
     * @param other The time to be compared with this time
     * @return boolean True if the received time is equal to this time
     */
    public boolean equals(Time2 other)
    {
        return minFromMidnight() == other.minFromMidnight();
    }

    /**
     * Check if this time is before a received time.
     * @param other The time to check if this time is before
     * @return boolean True if this time is before other time
    */
    public boolean before(Time2 other)
    {
        return minFromMidnight() < other.minFromMidnight();
    }

    /**
     * Check if this time is after a received time.
     * @param other The time to check if this time is before
     * @return boolean True if this time is after other time
    */
    public boolean after(Time2 other)
    {
        return other.before(this);
    }

    public int difference(Time2 other)
    {
        return minFromMidnight() - other.minFromMidnight();
    }

    /* Private methods */

    private static int calcMinFromMid(int hour, int minute)
    {
        return (hour * MINUTES_IN_HOUR) + minute;
    }

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