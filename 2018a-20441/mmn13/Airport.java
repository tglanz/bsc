/**
 * Represents a city's Airport.
 * The airport has up to a fixed amount of scheduled flights.
 */
public class Airport
{
    private static final int MAX_FLIGHTS = 200;

    private Flight[] _flightsSchedule;
    private int _noOfFlights;
    private String _airport;

    /**
     * Constructs an Airport object.
     * @param city The name city the airport is in.
     */
    public Airport(String city)
    {
        _airport = city;
        _flightsSchedule = new Flight[MAX_FLIGHTS];
        _noOfFlights = 0;
    }

    /**
     * Adds a flight to the schedued flights.
     * @param f The flight to add. nulls won't be added.
     * @return boolean true iff the flight was added.
     */
    public boolean addFlight(Flight f)
    {
        // argument validation
        if (f != null)
        {
            // make sure we have enough space in the flight schedule
            if (_noOfFlights < MAX_FLIGHTS)
            {
                // make sure this airport is in the flight's origin or destination
                if (_airport == f.getDestination() || _airport == f.getOrigin())
                {
                    _flightsSchedule[_noOfFlights++] = f;

                    return true;   
                }
            }
        }

        return false;
    }

    /**
     * Removes a scheduled flight.
     * @param f The scheduled flight to remove.
     * @return boolean true iff the flight was removed.
     */
    public boolean removeFlight(Flight f)
    {
        boolean didRemoveFlight = false;

        // argument validation
        if (f != null)
        {
            // loop through the flights to search the given one
            for (int idx = 0; idx < _noOfFlights && !didRemoveFlight; ++idx)
            {
                Flight currentFlight = _flightsSchedule[idx];

                if (f.equals(currentFlight))
                {
                    // by replacing the given flight with another we remove it from the schedule
                    _flightsSchedule[idx] = _flightsSchedule[_noOfFlights - 1];

                    // remove the flight which is now duplicated
                    _flightsSchedule[_noOfFlights - 1] = null;

                    // decrement the number of flights because we removed one
                    --_noOfFlights;

                    // set return value and break condition
                    didRemoveFlight = true;
                }
            }
        }

        return didRemoveFlight;
    }

    /**
     * Acquires the time of which the first flight to the place argument departures.
     * @param place The destination of the first flight to get the time of
     * @return Time1 The time of the first departure to the destination. Can be null if no flights were found.
     */
    public Time1 firstFlightFromDestination(String place)
    {
        Time1 minTime = null;

        // loop through the scheduled flights
        for (int idx = 0; idx < _noOfFlights; ++idx)
        {
            Flight currentFlight = _flightsSchedule[idx];

            // find the flights from place
            if (currentFlight.getOrigin() == place)
            {
                Time1 departure = currentFlight.getDeparture();

                // set minTime (return value) accordingly
                if (minTime == null)
                {
                    minTime = departure;
                }
                else if (departure.before(minTime))
                {
                    minTime = currentFlight.getDeparture();
                }
            }
        }

        return minTime;
    }

    /**
     * Return a string representation of this Airport.
     * @return String A representation of this Airport.
     */
    public String toString()
    {
        String content = "The flights for airport " + _airport + " today are:";

        for (int idx  = 0; idx < _noOfFlights; ++idx)
        {
            // append new line with the flight info
            content += "\n" + _flightsSchedule[idx].toString();
        }

        return content;
    }

    /**
     * Determines how many scheduled flights are full.
     * @return int Count of full scheduled flights.
     */
    public int howManyFullFlights()
    {
        int fullFlightsCount = 0;

        // iterate through the flights
        for (int idx = 0; idx < _noOfFlights; ++idx)
        {
            // filter the full flights
            if (_flightsSchedule[idx].getIsFull())
            {
                // increment the return value
                fullFlightsCount++;
            }
        }

        return fullFlightsCount;
    }

    /**
     * Determines how many flights departure and lands at city1 and city2 alternately.
     * @param city1 The first origin or destination
     * @param city2 The second origin or destination
     * @return int The count of flights that departures / lands at city1 and city2.
     */
    public int howManyFlightsBetween(String city1, String city2)
    {
        int flightsBetweenCount = 0;

        // iterate throught all of the flights
        for (int idx = 0; idx < _noOfFlights; ++idx)
        {
            Flight currentFlight = _flightsSchedule[idx];

            // if the flights flies from city1 to city2, or flies from city2 to city1
            if ((currentFlight.getOrigin() == city1 && currentFlight.getDestination() == city2) ||
                (currentFlight.getOrigin() == city2 && currentFlight.getDestination() == city1))
            {
                // increment the return value
                ++flightsBetweenCount;
            }
        }

        return flightsBetweenCount;
    }

    /**
     * Acquires the most popular destination.
     * @return String The destination with the most flights ocurrences.
     */
    public String mostPopularDestination()
    {
        // acquire all possible destinations
        String[] destinations = getDistinctDestinations();
        
        String popularDestination = "";
        int popularDestinationFlightsCount = 0;

        // for every destination
        for (int destinationIdx = 0; destinationIdx < destinations.length; ++destinationIdx)
        {
            String currentDestination = destinations[destinationIdx];
            int currentDestinationFlightsCount = 0;

            // count the number of flights to the current destination
            for (int flightIdx = 0; flightIdx < _noOfFlights; ++flightIdx)
            {
                // if the flight's destination is the same as the current destination
                if (currentDestination == _flightsSchedule[flightIdx].getDestination())
                {
                    // increment the count of flights to the current destination
                    ++currentDestinationFlightsCount;
                }
            }

            // if the current desinations flights count is bigger than
            // the current popular destination flights count
            if (currentDestinationFlightsCount > popularDestinationFlightsCount)
            {
                // the popular destination currently might only be the current destination
                popularDestinationFlightsCount = currentDestinationFlightsCount;
                popularDestination = currentDestination;
            }
        }

        return popularDestination;
    }

    /**
     * Acquires the flight which has the most expensive ticket.
     * @return Flight The flight with the most expensive ticket.
     */
    public Flight mostExpensiveTicket()
    {
        Flight mostExpensiveFlight = null;

        // iterate through the flights
        for (int idx = 0; idx < _noOfFlights; ++idx)
        {
            Flight currentFlight = _flightsSchedule[idx];

            // if the currently initialized most expensive flight is cheaper than the currently iterated flight
            if (mostExpensiveFlight == null || mostExpensiveFlight.isCheaper(currentFlight))
            {
                // set the return value
                mostExpensiveFlight = currentFlight;
            }
        }

        return mostExpensiveFlight;
    }

    /**
     * Acquires the Flight with the longest flight time.
     * @return Flight The flight with the longest flight time.
     */
    public Flight longestFlight()
    {
        Flight longestFlight = null;

        // iterate through the flights
        for (int idx = 0; idx < _noOfFlights; ++idx)
        {
            Flight currentFlight = _flightsSchedule[idx];

            // if the currently initialized longest flight is shorter than the current flight
            if (longestFlight == null || longestFlight.getFlightDuration() < currentFlight.getFlightDuration())
            {
                // set the return value
                longestFlight = currentFlight;
            }
        }

        return longestFlight;
    }

    // private

    /**
     * Acquires all of the destinations, without repetitions.
     */
    private String[] getDistinctDestinations()
    {
        int distinctDestinationsCount = 0;
        String[] distinctDestinations = new String[_noOfFlights];

        for (int flightIdx = 0; flightIdx < _noOfFlights; ++flightIdx)
        {
            String flightDestination = _flightsSchedule[flightIdx].getDestination();

            boolean isNewDestination = true;
            for (int destinationIdx = 0; !isNewDestination && destinationIdx < distinctDestinationsCount; ++destinationIdx)
            {
                if (flightDestination == distinctDestinations[destinationIdx])
                {
                    isNewDestination = false;
                }
            }

            if (isNewDestination)
            {
                distinctDestinations[distinctDestinationsCount] = flightDestination;
                ++distinctDestinationsCount;
            }
        }
        
        // normalize distinctDestinations to have no nulls
        // copy it to a new array with correct size
        String[] retVal = new String[distinctDestinationsCount];
        for (int idx = 0; idx < distinctDestinationsCount; ++idx)
        {
            retVal[idx] = distinctDestinations[idx];
        }

        return retVal;
    }
}