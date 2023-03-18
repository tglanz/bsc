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
    node = Node(problem.getStartState())

    # Initialize a new frontier, containing the initial state's
    # node as well as a set to hold the reached states.
    frontier = util.Stack()
    frontier.push(node)
    reached = set()

    # As long as the frontier is not empty, meaning we have additionals
    # states to investigate, investigate.
    while not frontier.isEmpty():
        node = frontier.pop()
        reached.add(node.state)

        # If we have reached the goal state we should trace back the
        # search tree and generate the actions that lead to the goal.
        if problem.isGoalState(node.state):
            return node.traceActions()
        
        # Get the successors of the current state and add them to the
        # frontier if they still haven't been investigated.
        for nextState, action, cost in problem.getSuccessors(node.state):
            childNode = Node(nextState, node, action, cost)
            if nextState not in reached:
                frontier.push(childNode)
    
    # No solution was found
    return None

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    node = Node(problem.getStartState())

    # Initialize a new frontier, containing the initial state's
    # node as well as a set to hold the reached states.
    # We will also check if states are in the frontiers. To do this efficiently
    # we will store a set of the frontier states (vs the node).
    frontier = util.Queue()
    frontier.push(node)
    frontierStates = set([node.state])
    reached = set()

    # As long as the frontier is not empty, meaning we have additionals
    # states to investigate, investigate.
    while not frontier.isEmpty():
        node = frontier.pop()
        frontierStates.remove(node.state)
        reached.add(node.state)

        # If we have reached the goal state we should trace back the
        # search tree and generate the actions that lead to the goal.
        if problem.isGoalState(node.state):
            return node.traceActions()
        
        # Get the successors of the current state and add them to the
        # frontier if they still haven't been investigated.
        for nextState, action, cost in problem.getSuccessors(node.state):
            childNode = Node(nextState, node, action, cost)
            if nextState not in reached and nextState not in frontierStates:
                frontier.push(childNode)
                frontierStates.add(nextState)
        
    # No solution was found
    return None

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    node = Node(problem.getStartState(), cost=0)

    # Initialize a new frontier, containing the initial state's
    # node as well as a set to hold the reached states.
    frontier = util.PriorityQueueWithFunction(lambda node: node.cost)
    frontier.push(node)
    reached = set()

    # In UCS we need access to the Node according to state in order to check
    # whether it provides a cheaper route, and if so we should replace it.
    frontierStates = {node.state: node}

    # Get the successors of the current state and add them to the
        # frontier if they still haven't been investigated.
    while not frontier.isEmpty():
        node = frontier.pop()
        del frontierStates[node.state]
        reached.add(node.state)

        # If we have reached the goal state we should trace back the
        # search tree and generate the actions that lead to the goal.
        if problem.isGoalState(node.state):
            return node.traceActions()
        
        # Get the successors of the current state and add them to the
        # frontier if they still haven't been investigated.
        for nextState, action, cost in problem.getSuccessors(node.state):
            childNode = Node(nextState, node, action, node.cost + cost)

            # If the state hasn't been reached yet and it is not in the frontier
            # we must surely investigate it in the future.
            if nextState not in reached and nextState not in frontierStates:
                frontier.push(childNode)
                frontierStates[nextState] = childNode
            elif nextState in frontierStates and frontierStates[nextState].isMoreExpensive(childNode):
                # In UCS we should also check, even if a child is already in the frontier,
                # perhaps the current route to it is cheaper then before? if so, we need to replace it!
                frontierStates[nextState].copyFrom(childNode)
                frontier.update(frontierStates[nextState], childNode.cost)
    
    # No solution was found
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    node = Node(problem.getStartState(), cost=0)

    # Initialize a new frontier, containing the initial state as well as the reached state containing the same state.
    # The aStartSearch algorithm is very similiar to UCS with the difference of the prioritization function - 
    # aStart uses cost + heuristic where UCS only uses cost.
    frontier = util.PriorityQueueWithFunction(lambda node: node.cost + heuristic(node.state, problem))
    frontier.push(node)
    reached = set()

    # We need access to the Node according to state in order to check
    # whether it provides a cheaper route, and if so we should replace it.
    frontierStates = {node.state: node}

    # As long as the frontier is not empty, meaning we have additionals
    # states to investigate, investigate.
    while not frontier.isEmpty():
        node = frontier.pop()
        del frontierStates[node.state]
        reached.add(node.state)

        # If we have reached the goal state we should trace back the
        # search tree and generate the actions that lead to the goal.
        if problem.isGoalState(node.state):
            return node.traceActions()
        
        # Get the successors of the current state and add them to the
        # frontier if they still haven't been investigated.
        for nextState, action, cost in problem.getSuccessors(node.state):
            childNode = Node(nextState, node, action, node.cost + cost)

            # If the state hasn't been reached yet and it is not in the frontier
            # we must surely investigate it in the future.
            if nextState not in reached and nextState not in frontierStates:
                frontier.push(childNode)
                frontierStates[nextState] = childNode
            elif nextState in frontierStates and frontierStates[nextState].isMoreExpensive(childNode):
                # We should also check, even if a child is already in the frontier,
                # perhaps the current route to it is cheaper then before? if so, we need to replace it!
                frontierStates[nextState].copyFrom(childNode)
                frontier.update(frontierStates[nextState], childNode.cost)
    
    # No solution was found.
    return None

class Node:
    """Node is an implicit datastructure which represents a node in the SearchTree/SearchGraph
    
    Parameters
      state: The state this node holds
      previous: The ancestor of this node
      action: The action used to transition to this state from the ancestor
      cost: The cost of the action
    """
    def __init__(self, state, previous=None, action=None, cost=None):
        self.state = state
        self.previous = previous
        self.action = action
        self.cost = cost
    
    def __iter__(self):
        """Used for convinient"""
        return self.previousState, self.currentState, self.action, self.cost

    def iterateAncestors(self):
        """Iterate all of the ancestors in order"""
        yield self
        previous = self.previous
        while previous is not None:
            yield previous
            previous = previous.previous
    
    def traceActions(self):
        """Create a list of actions used to reach this state from the start"""
        return list(reversed([
            ancestor.action
            for ancestor in self.iterateAncestors()
            if ancestor.action is not None
        ]))
    
    def isMoreExpensive(self, otherNode):
        """Checks whether this node's cost is more exensive the otherNode
        
        Parameters
          otherNode: The Node to compare the cost with.
        """
        return self.cost > otherNode.cost

    def copyFrom(self, otherNode):
        """Copy the data from another node
        
        Use this function to replace data from another node but keep the same reference.

        Parameters:
          otherNode: The Node to copy data from.
        """
        self.state = otherNode.state
        self.previous = otherNode.previous
        self.action = otherNode.action
        self.cost = otherNode.cost

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
