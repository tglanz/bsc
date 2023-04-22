# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Calculate the bias according to food.
        # Pacman's goal is to eat as much as possible as fast as possible.
        # It makes sense to give higher evaluation to actions that brings him closer to a food.
        # Technically, small distances should translate to higher values - we calculate those as inverses.
        foodBias = 0
        foodPositions = newFood.asList()
        if foodPositions:
            newMinDistance = min(map(lambda foodPos: util.manhattanDistance(newPos, foodPos), foodPositions))
            foodBias = 1 / newMinDistance

        # On the contrary to food, Pacman wants to avoid ghosts.
        ghostFactor = min(map(lambda ghostState: util.manhattanDistance(newPos, ghostState.getPosition()), newGhostStates), default=1)
        if ghostFactor > 4:
            ghostFactor = 4
        
        return successorGameState.getScore() + (ghostFactor * foodBias)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Parameters:
          gameState: MultiagentTreeState
        """
        score, action = self.maxValue(gameState, 0)
        return action

    def isTerminal(self, state):
        return any((state.isWin(), state.isLose()))

    def maxValue(self, state, depth):
        """
        Maximize the utility over legal actions.

        Returns the utility and the action made.

        Parameters
            state: MultiagentTreeState - the current state the search is investigating
            depth: int indicating the current depth in the search
        """
        if self.isTerminal(state) or depth >= self.depth:
            return self.evaluationFunction(state), None

        score, action = None, None
        agent = 0
        for candidateAction in state.getLegalActions(agent):
            candidateState = state.generateSuccessor(agent, candidateAction)
            candidateScore, _ = self.minValue(candidateState, agent + 1, depth)
            if score is None or candidateScore > score:
                score = candidateScore
                action = candidateAction
        
        return score, action
    
    def minValue(self, state, agent, depth):
        """
        Minimize the utility over legal actions.

        Returns the utility and the action made.

        Parameters
            state: MultiagentTreeState - the current state the search is investigating
            agent: int indicating the agent performing the search.
                       expected to be the agent index of some ghost since pacman only maximize.
            depth: int indicating the current depth in the search
        """
        if agent == 0 or agent >= state.getNumAgents():
            raise f"Invalid agent {agent}. Expected to be a ghost"

        if self.isTerminal(state):
            return self.evaluationFunction(state), None

        score, action = None, None
        for candidateAction in state.getLegalActions(agent):
            candidateState = state.generateSuccessor(agent, candidateAction)

            # last ghost
            if agent == state.getNumAgents() - 1:
                candidateScore, _ = self.maxValue(candidateState, depth + 1)
            else:
                candidateScore, _ = self.minValue(candidateState, agent + 1, depth)

            if score is None or candidateScore < score:
                score = candidateScore
                action = candidateAction
        
        return score, action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Parameters:
          gameState: MultiagentTreeState
        """
        score, action = self.maxValue(gameState, 0)
        return action

    def isTerminal(self, state):
        return any((state.isWin(), state.isLose()))

    def maxValue(self, state, depth):
        """
        Maximize the utility over legal actions.

        Returns the utility and the action made.

        Parameters
            state: MultiagentTreeState - the current state the search is investigating
            depth: int indicating the current depth in the search
        """
        if self.isTerminal(state) or depth >= self.depth:
            return self.evaluationFunction(state), None

        score, action = None, None
        agent = 0
        for candidateAction in state.getLegalActions(agent):
            candidateState = state.generateSuccessor(agent, candidateAction)
            candidateScore, _ = self.minValue(candidateState, depth, agent + 1)
            if score is None or candidateScore > score:
                score = candidateScore
                action = candidateAction
        
        return score, action
    
    def minValue(self, state, depth, agent):
        """
        Minimize the utility over legal actions.

        Returns the utility and the action made.

        Parameters
            state: MultiagentTreeState - the current state the search is investigating
            agent: int indicating the agent performing the search.
                       expected to be the agent index of some ghost since pacman only maximize.
            depth: int indicating the current depth in the search
        """
        if agent == 0 or agent >= state.getNumAgents():
            raise f"Invalid agent {agent}. Expected to be a ghost"

        if self.isTerminal(state):
            return self.evaluationFunction(state), None

        score, action = None, None
        for candidateAction in state.getLegalActions(agent):
            candidateState = state.generateSuccessor(agent, candidateAction)

            # last ghost
            if agent == state.getNumAgents() - 1:
                candidateScore, _ = self.maxValue(candidateState, depth + 1)
            else:
                candidateScore, _ = self.minValue(candidateState, depth, agent + 1)

            if score is None or candidateScore < score:
                score = candidateScore
                action = candidateAction
        
        return score, action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
