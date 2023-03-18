# 20551, MMN 11

Author: Tal Glanzman
Date: 18/03/2023

To run autograder for all questions:

    for i in $(seq 1 8); do python autograder.py -q q${i} | grep "### Question q${i}"; done

## Q6

The cornersHeuristic is implemented as the manhatten distance from the farthest (same metric) unvisited corner.

It is admissable because a goal state can be reach only if pacman already been in all corners, the farthest specifically. Because the manhatten distance is the shortest distance, the path to the goal is no less than the manhatten distance to the farthest corner.

## Q7

The foodHeuristic is implemented as the real distance (provided to use by the function mazeDistance) from pacman to the farthest food.

Admissable reasoning is simliar to the admissability reasoning of cornersHeuristic.

We chose the mazeDistance instead of manhattenDistance because mazeDistance >= manhattenDistance - We achieve an admissable and dominant heuristic (dominant with respect to a hueristic that was implemented using the manhatten distance).