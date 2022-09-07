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


from email.errors import MisplacedEnvelopeHeaderDefect
from util import manhattanDistance
from game import Directions
from collections import deque
import random, util
import copy

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        foodScore = successorGameState.getScore()
        foodList = newFood.asList()
        walls = copy.deepcopy(currentGameState.getWalls())

        if successorGameState.isWin():
            foodScore = float('inf')
            return foodScore

        for i in newGhostStates:
            if newPos == i.getPosition():
                foodScore = float('-inf')
                return foodScore
        
        q = deque([(newPos, 0)])
        dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while(True):
            front = q.popleft()

            for i in dir:
                if (walls[front[0][0]+i[0]][front[0][1]+i[1]]):
                    continue

                q.append(((front[0][0]+i[0], front[0][1]+i[1]), front[1]+1))
                walls[front[0][0]+i[0]][front[0][1]+i[1]] = True

            if newFood[front[0][0]][front[0][1]]:
                foodScore += 10/front[1]
                break

        if action == "Stop":
            foodScore = float('-inf')
        
        return foodScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        stateList = [(gameState.generateSuccessor(0, action), action) for action in actions]
        maxValue = float('-inf')
        result = actions[0]

        for state, action in stateList:
            x = self.helper(state, 1, 0)
            if x > maxValue:
                maxValue = x
                result = action
                
        return result

    def helper(self, gameState: GameState, agentIndex, depth):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth += 1
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxvalue(gameState, agentIndex, depth)
        else:
            return self.minvalue(gameState, agentIndex, depth)

    def maxvalue(self, gameState: GameState, agentIndex, depth):
        m = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            m = max(m, self.helper(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth))
        return m

    def minvalue(self, gameState: GameState, agentIndex, depth):
        m = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            m = min(m, self.helper(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth))
        return m


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        stateList = [(gameState.generateSuccessor(0, action), action) for action in actions]
        maxValue = float('-inf')
        result = actions[0]
        alpha = float('-inf')
        beta = float('inf')

        for state, action in stateList:
            x = self.helper(state, 1, 0, alpha, beta)
            if x > maxValue:
                maxValue = x
                result = action
                alpha = x
                
        return result

    def helper(self, gameState: GameState, agentIndex, depth, alpha, beta):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth += 1
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxvalue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minvalue(gameState, agentIndex, depth, alpha, beta)

    def maxvalue(self, gameState: GameState, agentIndex, depth, alpha, beta):
        m = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            m = max(m, self.helper(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, alpha, beta))
            if m > beta:
                return m
            alpha = max(alpha, m)
        return m

    def minvalue(self, gameState: GameState, agentIndex, depth, alpha, beta):
        m = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            m = min(m, self.helper(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, alpha, beta))
            if m < alpha:
                return m
            beta = min(beta, m)
        return m


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        stateList = [(gameState.generateSuccessor(0, action), action) for action in actions]
        maxValue = float('-inf')
        result = actions[0]

        for state, action in stateList:
            x = self.helper(state, 1, 0)
            if x > maxValue:
                maxValue = x
                result = action
                
        return result

    def helper(self, gameState: GameState, agentIndex, depth):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth += 1
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxvalue(gameState, agentIndex, depth)
        else:
            return self.minvalue(gameState, agentIndex, depth)

    def maxvalue(self, gameState: GameState, agentIndex, depth):
        m = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            m = max(m, self.helper(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth))
        return m

    def minvalue(self, gameState: GameState, agentIndex, depth):
        m = 0
        for action in gameState.getLegalActions(agentIndex):
            m += self.helper(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth)
        return m/len(gameState.getLegalActions(agentIndex))


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    foodScore = successorGameState.getScore()
    foodList = newFood.asList()
    walls = copy.deepcopy(currentGameState.getWalls())

    if successorGameState.isWin():
        foodScore = float('inf')
        return foodScore

    for i in newGhostStates:
        if newPos == i.getPosition():
            foodScore = float('-inf')
            return foodScore
    
    q = deque([(newPos, 0)])
    dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while(True):
        front = q.popleft()

        for i in dir:
            if (walls[front[0][0]+i[0]][front[0][1]+i[1]]):
                continue

            q.append(((front[0][0]+i[0], front[0][1]+i[1]), front[1]+1))
            walls[front[0][0]+i[0]][front[0][1]+i[1]] = True

        if newFood[front[0][0]][front[0][1]]:
            foodScore += 10/front[1]
            break
    
    walls = copy.deepcopy(currentGameState.getWalls())
    q = deque([(newPos, 0)])
    dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while(True and len(q) != 0):
        front = q.popleft()

        for i in dir:
            if (walls[front[0][0]+i[0]][front[0][1]+i[1]]):
                continue

            q.append(((front[0][0]+i[0], front[0][1]+i[1]), front[1]+1))
            walls[front[0][0]+i[0]][front[0][1]+i[1]] = True

        if (front[0][0], front[0][1]) in currentGameState.getGhostPositions():
            foodScore -= 20/front[1]
            break

    return foodScore

# Abbreviation
better = betterEvaluationFunction
