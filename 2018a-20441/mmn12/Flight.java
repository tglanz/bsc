public class Flight
{
    private static final int MINUTES_IN_HOUR = 60;

    private static final int MAX_CAPACITY = 250;
    private static final int MIN_CAPACITY = 0;
    private static final int MIN_FLIGHT_DURATION = 0;
    private static final int MIN_PRICE = 0;

    private String _origin;
    private String _destination;
    private Time1 _departure;
    private int _flightDuration;
    private int _noOfPassengers;
    private boolean _isFull;
    private int _price;

    /**
     * Copy constructor for a Flight object. Construct a Flight object with the same attributes as another Flight object.
     * @param other The Flight object from which to construct the new Flight.
     */
    public Flight(Flight other)
    {
        _origin = other._origin;
        _destination = other._destination;
        _departure = other._departure;
        _flightDuration = other._flightDuration;
        _noOfPassengers = other._noOfPassengers;
        _isFull = other._isFull;
        _price = other._price;
    }

    /**
     * Constructor for a Flight object.
     * If the number of passengers exceeds the maximum capacity the number of passengers
     * will be set to the maxmum capacity If the number of passengers is negative the number of passengers will be set to zero.
     * If the flight duration is negative the flight duration will be set to zero.
     * If the price is negative the price will be set to zero.
     * @param dest The city the flight lands at.
     * @param origin The city the flight leaves from.
     * @param depHour the departure hour (should be between 0-23).
     * @param depMinute The departure minute (should be between 0-59).
     * @param durTimeMinutes The duration time in minutes(should not be negative).
     * @param noOfPass The number of passengers (should be between 0-maximum capacity).
     * @param price The price for a ticket
     */
    public Flight(String origin,
        String dest,
        int depHour,
        int depMinute,
        int durTimeMinutes,
        int noOfPass,
        int price)
    {
        // Validity checks
        if (durTimeMinutes < MIN_FLIGHT_DURATION)
        {
            durTimeMinutes = MIN_FLIGHT_DURATION;
        }

        if (noOfPass > MAX_CAPACITY)
        {
            noOfPass = MAX_CAPACITY;
        }

        if (noOfPass < MIN_CAPACITY)
        {
            noOfPass = MIN_CAPACITY;
        }

        if (price < MIN_PRICE)
        {
            price = MIN_PRICE;
        }

        // Set the fields
        _origin = origin;
        _destination = dest;
        _departure = new Time1(depHour, depMinute);
        _flightDuration = durTimeMinutes;
        _noOfPassengers = noOfPass;
        _price = price;
        _isFull = _noOfPassengers == MAX_CAPACITY;
    }

    /**
     * Returns the flight departure time.
     * @return Time1 A copy of the flight departure time.
     */
    public Time1 getDeparture()
    {
        // Make a copy, don't return the same reference
        return new Time1(_departure);
    }

    /**
     * Returns the flight destination.
     * @return String The flight destination.
     */
    public String getDestination()
    {
        return _destination;
    }

    /**
     * Returns he flight duration time in minutes.
     * @return int The flight duration.
     */
    public int getFlightDuration()
    {
        return _flightDuration;
    }

    /**
     * Returns whether the flight is full or not.
     * @return boolean True if the flight is full.
     */
    public boolean getIsFull()
    {
        return _isFull;
    }

    /**
     * Returns the number of passengers on the flight.
     * @return int The number of passengers.
     */
    public int getNoOfPassengers()
    {
        return _noOfPassengers;
    }    

    /**
     * Returns the flight origin.
     * @return String The flight origin.
     */
    public String getOrigin()
    {
        return _origin;
    }

    /**
     * Returns the price of the flight.
     * @return int The price.
     */
    public int getPrice()
    {
        return _price;
    }

    /**
     * Changes the flight's departure time.
     * @param departureTime The flight's departure time.
     */
    public void setDeparture(Time1 departureTime)
    {
        _departure = departureTime;
    }

    /**
     * Changes the flight's destination.
     * @param dest The flight's new destination.
     */    
    public void setDestination(String dest)
    {
        _destination = dest;
    }

    /**
     * Changes the flight's duration time.
     * If the parameter is negative the duration time will remain unchanged.
     * @param durTimeMinutes The flight's new duration time.
     */
    public void setFlightDuration(int durTimeMinutes)
    {
        // Make sure the argument is valid
        if (MIN_FLIGHT_DURATION <= durTimeMinutes)
        {
            _flightDuration = durTimeMinutes;
        }
    }

    /**
     * Changes the number of passengers.
     * If the parameter is negative or larger than the maximum capacity the number of passengers will remain unchanged.
     * @param noOfPass The new number of passengers.
     */    
    public void setNoOfPassengers(int noOfPass)
    {
        // Make sure the argument is valid
        if (MIN_CAPACITY <= noOfPass && noOfPass <= MAX_CAPACITY)
        {
            _noOfPassengers = noOfPass;
        }
    }

    /**
     * Changes the flight's origin.
     * @param origin The flight's new origin.
     */
    public void setOrigin(String origin)
    {
        _origin = origin;
    }

    /**
     * Changes the flight price.
     * If the parameter is negative the price will remain unchanged.
     * @param price The new price.
     */    
    public void setPrice(int price)
    {
        // Make sure the argument is valid
        if (MIN_PRICE <= price)
        {
            _price = price;
        }
    }

    /**
     * Check if the received flight is equal to this flight. Flights are considered equal if the origin, destination and departure times are the same.
     * @param other The flight to be compared with this flight.
     * @return boolean True if the received flight is equal to this flight.
     */
    public boolean equals(Flight other)
    {
        // Compare all of the fields

        return
            _origin == other._origin &&
            _destination == other._destination &&
            _departure == other._departure &&
            _flightDuration == other._flightDuration &&
            _noOfPassengers == other._noOfPassengers &&
            _isFull == other._isFull &&
            _price == other._price;
    }

    /**
     * Returns the arrival time of the flight 
     * @return Time1 The arrival time of this flight.
     */
    public Time1 getArrivalTime()
    {
        int departureTimeInMinutes =
            getDeparture().getHour() * MINUTES_IN_HOUR + getDeparture().getMinute();

        int arrivalTimeInMinutes = departureTimeInMinutes + _flightDuration;

        int arrivalHour = arrivalTimeInMinutes / MINUTES_IN_HOUR;
        int arrivalMinute = arrivalTimeInMinutes % MINUTES_IN_HOUR;

        return new Time1(arrivalHour, arrivalMinute);
    }

    /**
     * Add passengers to this flight.
     * If the number of passengers exceeds he maximum capacity, no passengers are added and alse is returned.
     * If the flight becomes full, the boolean attribute describing whether the flight if full becomes true.
     * Assume parameter is positive.
     * @param num The number of passengers to be added to this flight.
     * @return boolean True if the passengers were added to the flight.
     */
    public boolean addPassengers(int num)
    {
        // Check if the number of passengers after addition is no more then allowed
        if (_noOfPassengers + num <= MAX_CAPACITY)
        {
            _noOfPassengers += num;

            if (_noOfPassengers == MAX_CAPACITY)
            {
                _isFull = true;
            }

            return true;
        }

        return false;
    }

    /**
     * Check if this flight is cheaper than another flight.
     * @param other The flight whose price is to be compared with this flight's price.
     * @return boolean True if this flight is cheaper than the received flight .
     */
    public boolean isCheaper(Flight other)
    {
        return getPrice() < other.getPrice();
    }

    /**
     * Calculate the total price of the flight.
     * @return int The total price of the flight.
     */
    public int totalPrice()
    {
        return getNoOfPassengers() * getPrice();
    }

    /**
     * Check if this flight lands before another flight.
     * Note - the flights may land on different days, the method checks which flight lands first.
     * @param other The flight whose arrival time to be compared with this flight's arrival time.
     * @return boolean True if this flight arrives before the received flight.
     */
    public boolean landsEarlier(Flight other)
    {
        return getArrivalTime().before(other.getArrivalTime());
    }

    /**
     * Return a string representation of this flight
     * (for example: "Flight from London to Paris departs at 09:24.Flight is full.").
     * @return String String representation of this flight (for example: "Flight from London to Paris departs at 09:24.Flight is full.").
     */
    public String toString()
    {
        String retVal = "Flight from " + _origin + " to " + _destination + " departs at " + _departure + ". ";

        if (_isFull)
        {
            retVal += "Flight is full.";
        }
        else
        {
            retVal += "Flight is not full.";
        }

        return retVal;
    }
}