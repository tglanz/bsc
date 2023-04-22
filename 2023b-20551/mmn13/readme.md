# MMN13, 20551

Author: Tal Glanzman
Date: 22/04/2023

In MMN11 I got a remark that the readme is lacking - I hope this is better.

## Common Commands

Start a game of pacman

    python pacman.py

We can specify which agent to use for Pacman by providing the ```-p/--pacman``` argument. By default, it is set to _KeyboardAgent_. For example, to run set the _RefelxAgent_ run:

    python pacman.py -p ReflexAgent

We can also specify the layout using the ```-l/--layout``` argument. By default, it is set to _mediumClassic_. Example layouts:

- _testClassic_: The most trivial layout - Used mainly for testing purposes.

In order to specify the number of ghosts use the ```-k/--numghosts``` argument. This argument is 4 by default.

Some examples:

    python pacman.py -p ReflexAgent -l testClassic -k 0
    python pacman.py -p ReflexAgent -l testClassic -k 1
    python pacman.py -p ReflexAgent -l mediumClassic -k 1
    python pacman.py -p ReflexAgent -l mediumClassic -k 2

## Q1, Designing a better evaluation function

Given the current game state and an action, the ```ReflexAgent::evaluationFunction``` returns a number. The higher the number, the better the given action is evaluated.

The existing implementation did not take into account the food positions nor the ghost positions! This is the following implications:
- Because the evaluation ignored the food position (up to the score of the next state), Pacman has no incentive to complete the level fast. 
- Because the evaluation ignored the ghost positions (up to the score of the next state), Pacman could easily collide with a ghost and died.

From the said above, we can assume a better evaluation function which takes the relevant information into account. 

The closer Pacman gets to a food position due to the current action the higher the evaluation should be. We will track this relation using the ```foodBias``` variable. Note that this is an inverse relation - we chose to implement it as ```foodBias = 1 / minDistanceToFood```.

Pacman likes safety - The farther pacman is from ghosts the better. More accurately, the closer Pacman is to a ghost, the less he cares about food. We will model this as a multiplicatve relation - ```ghostFactor * foodBias``` where ```ghostFactor``` is the minimum distance to any ghost in the level.

Empiricaly, the bigger the level the higher the chance that Pacman is no where near a ghost. Also taking into account the fact that ghosts has the same speed as Pacman and can't "catch up to him", we can have Pacman forget about the ghost from a certain distance. We chose this constant distance to be 4. The reason that a lesser distance can cause Pacman to collide with a ghost is that 3 is the distance where Pacman doesn't have the time to change directions can escape the ghosts and get away with it alive.

Finally, we give the evaluation function ```evaluation = nextStateScore + (ghostFactor * foodBias)```.

## Q2, Minimiax with ghosts alliance

Here we have more than two agents. Specifically, given the number of agents N, pacman is the agent 0 and the ghosts are agents 1, 2, ..., N-1.

The concept remains similiar to two player minimax with the following neuances: Pacman is equivalent to the MAX player - he tries to maximize the utility from all the available actions he can make. On the opposite, the ghosts are equivalent to the MIN player - all of them try to minimize the utility among all of their available actions.

In our model, the game is turn based - Agents make their move one after another. 

From the above, we can understand the modified search tree: Each level is composed of N sub-levels. The first sub level is the states that Pacman tries to maximize the actions from. The following N-1 sub levels are the states that the ghosts 1, 2, ..., N-1 accordingly try to minimize.

Also note that we need to limit the depth. This is simple, just keep track of the current depth and remember that a depth is with respect to the sub-levels discussed above. Meaning, a search tree of level K is at depth K%N.

From the above, the modifications to the algorithm that can be found in the book is that the ```Min-Value``` function should be performed on all ghosts, one by one. Only after the last ghost, a call to ```Max-Value``` (for pacman's turn) should be made.

## Q3, Multi-Agent Alpha-Beta pruning

In reality, the fact that there are multiple MIN players (the ghosts) doesn't have much affect on the implementation. Like the standard algorith, update the alpha and beta parameters while traversing down the search, update the value going up.

All I really did was copy paste the ```minValue``` and ```maxValue``` I implemented in ```MinimaxAgent``` and added alpha-beta parameters as instructed in the Mamans notebook.

## Q4, Expectimax

Pacman logic remains the same - he still wants to make the best action possible (maximize utility).

The ghosts return the average utility instead of their minimum. This manifests the fact that they move randomly.

## Q5, Designing a better evaluation function

The process of designing the better evaluation function was part intuition and part trial-and-error.

If we examine the game of pacman at a broad level pacman's main interests are:

- Not dying
- Eating food
- Avoid stalling

We designed the evaluation function as a dot product of a priority vector and an elements vector measuring different criteria in the game. Such approach we can fine tune different aspects of the function - which is where the trial-and-error came in.

The components of the evaluation are described below:

Not Dying / Safety: In order for pacman to try and avoid death, we measured the (manhanttan) distance from him to the nearest ghost. The closer he is, the more dangerous it is. Meaning, the higher the distance, the better the evaluation should be since it indicates safety.

Eating food: All things aside, Pacman should walk towards food at any given time since his win condition is by clearing the level. The negative of the closest food to him is a function that increases as he get closer to food.

Avoid stalling: The state score encodes penalty for time spent. Hence, we use the score of the state as part of the evaluation - Less time spent, the higher the evaluation.

Another component which has an impact is the "Freedom of movement". If the ghosts are in the scared state, Pacman can move as freely as he likes. Therefore, it is wise to eat the capsules.

Also, in the game of Pacman, eating ghosts increase the score. Therefore, we added to the evaluation the negative of the number of the ghosts on the board. Practically, and perhaps unintuively, it didn't seem to have much of an effect. Perhaps it's because that unlike a normal game of pacman the ghosts spawn immediately - taking away some of the benefits? Perhaps maybe also because the ghosts spawn afterwards, the number of ghosts increases again causing some rebound effect on the evaluation?