# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    return SearchAlgorithm(util.Stack).solve(problem)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return SearchAlgorithm(util.Queue).solve(problem)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
class SearchAlgorithm:
    class Node:
        def __init__(self, state, previous=None, action=None, cost=None):
            self.state = state
            self.previous = previous
            self.action = action
            self.cost = cost
        
        def __iter__(self):
            return self.previousState, self.currentState, self.action, self.cost

        def iterateAncestors(self):
            yield self
            previous = self.previous
            while previous is not None:
                yield previous
                previous = previous.previous

    """
    SearchAlgorithm is the algorithm that perform the searches.

    Every search algorithm, be it DFS, BFS, A* and so on, differ only by the strategy
    at which it chooses the next nodes to explore - This is managed according to the fringe's
    datastructure which is given as a dependency. 

    Practically this is a "hosting" algorithm that should be
    injected with different strategies for execution - a.k.a strategy pattern.

    Parameters
      frontierConstructor: A function to create the frontier to be used by the algorihm.
                Usually it is one of the datastructures in util.py.
                Every frontier should support the following interface:
                - push(element: Any)
                - pop() -> Any
                - isEmpty() -> Boolean

    Example
      algorithm = SearchAlgorithm(util.Stack)
      solution = algorithm.solve(problem)
    """
    def __init__(self, frontierConstructor):
        self.createFrontier = frontierConstructor
    
    def solve(self, problem: SearchProblem):
        startState = problem.getStartState()

        # initialize a new frontier, containing the initial state
        # as well as the reached state containing the same state.
        frontier = self.createFrontier()
        frontier.push(SearchAlgorithm.Node(startState))
        reached = set([startState])

        # as long as the frontier is not empty, meaning we have additionals
        # states to investigate, investigate.
        while not frontier.isEmpty():
            node = frontier.pop()
            state = node.state

            # if we have reached the goal state we should trace back the
            # search tree and generate the actions that lead to the goal.
            if problem.isGoalState(state):
                actions = [ancestor.action
                    for ancestor
                    in node.iterateAncestors()
                    if ancestor.action is not None]
                return list(reversed(actions))
            
            # get the successors of the current state and add them to the
            # frontier if they still haven't been investigated.
            for nextState, action, cost in problem.getSuccessors(state):
                if nextState not in reached:
                    successor = SearchAlgorithm.Node(nextState, node, action, cost)
                    frontier.push(successor)
                    
                    # we are done investigating node. add it to reached.
                    reached.add(nextState)
            
        
        # No solution was found.
        return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
