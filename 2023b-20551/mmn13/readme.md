# MMN13

Author: Tal Glanzman

## Common Commands

Start a game of pacman

    python pacman.py

We can specify which agent to use for Pacman by providing the ```-p/--pacman``` argument. By default, it is set to _KeyboardAgent_. For example, to run set the _RefelxAgent_ run:

    python pacman.py -p ReflexAgent

We can also specify the layout using the ```-l/--layout``` argument. By default, it is set to _mediumClassic_. Example layouts:

- _testClassic_: The most trivial layout - Used mainly for testing purposes.

In order to specify the number of ghosts use the ```-k/--numghosts``` argument. This argument is 4 by default.