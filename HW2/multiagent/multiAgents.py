

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        averageDistance = 0

        distanceToFood = map(lambda s: 1.0 / manhattanDistance(newPos, s), newFood.asList())
        if len(distanceToFood) != 0:
            averageDistance = float(sum(distanceToFood)) / float(len(distanceToFood))

        ghostScore = 0
        if newScaredTimes:
            distanceToGhost = map(lambda g: manhattanDistance(newPos, g.getPosition()), newGhostStates)
            smallestGhostDistance = min(distanceToGhost)
            if smallestGhostDistance < 2:
                ghostScore = -9999999

        return successorGameState.getScore() + averageDistance + ghostScore

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

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        # For the Ghosts
        def mini(state, ghost, depth):
            """
            :param state - a given state of the game
            :param ghost - the agent index of the ghost maxi is being calculated
                            for
            :param depth - the current depth calculated
            :return list - a list with two elements, the utility of that state,
                            and the action taken to get there
            """
            # moves for ghost
            legalMoves = state.getLegalActions(ghost)

            if state.isLose() or not legalMoves or self.depth < depth:
                return [self.evaluationFunction(state), None]

            # if there are still ghosts to calculate
            total_ghosts = gameState.getNumAgents() - 1
            if ghost != total_ghosts:
                utilities = []
                # get successors of actions
                for action in legalMoves:
                    suc = state.generateSuccessor(ghost, action)
                    utilities.append( (mini(suc, ghost + 1, depth), action) )
            # else if we are on the last ghost
            else:
                utilities = []
                # get successors of actions
                for action in legalMoves:
                    suc = state.generateSuccessor(ghost, action)
                    utilities.append( (maxi(suc, depth + 1), action) )


            return min(utilities)

        # Pacman
        def maxi(state, depth):
            """
            :param state - a given state of the game
            :param depth - the current depth calculated
            :return list - a list with two elements, the utility of that state,
                            and the action taken to get there
            """
            # moves for pacman
            legalMoves = state.getLegalActions(0)
            if  state.isWin() or not legalMoves or self.depth < depth:
                return [self.evaluationFunction(state), None]

            utilities = []
            # get successors of actions
            for action in legalMoves:
                suc = state.generateSuccessor(0, action)
                utilities.append( (mini(suc, 1, depth), action) )

            return max(utilities)

        m = maxi(gameState, 1)
        utility, action = m[0], m[1]
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # For the Ghosts
        def mini(state, ghost, depth, a, b):
            """
            :param state - a given state of the game
            :param ghost - the agent index of the ghost maxi is being calculated
                            for
            :param depth - the current depth calculated
            :return list - a list with two elements, the utility of that state,
                            and the action taken to get there
            """
            # moves for ghost
            legalMoves = state.getLegalActions(ghost)

            if state.isLose() or not legalMoves or self.depth < depth:
                return [self.evaluationFunction(state), None]

            v = float('inf')
            chosenAction = None
            total_ghosts = state.getNumAgents() - 1
            for action in legalMoves:

                if ghost != total_ghosts:
                    suc = state.generateSuccessor(ghost, action)
                    utility = mini(suc, ghost + 1, depth, a, b)

                else:
                    suc = state.generateSuccessor(ghost, action)
                    utility = maxi(suc, depth + 1, a, b)

                if utility[0] < v:
                    v = utility[0]
                    chosenAction = action
                if v < a:
                    return [v, chosenAction]

                b = min(b, v)
            return [v, chosenAction]


        # Pacman
        def maxi(state, depth, a, b):
            """
            :param state - a given state of the game
            :param depth - the current depth calculated
            :return list - a list with two elements, the utility of that state,
                            and the action taken to get there
            """
            # moves for pacman
            legalMoves = state.getLegalActions(0)
            if  state.isWin() or not legalMoves or self.depth < depth:
                return [self.evaluationFunction(state), None]

            v = float('-inf')
            chosenAction = None
            # get successors of actions
            for action in legalMoves:
                suc = state.generateSuccessor(0, action)
                utility =  mini(suc, 1, depth, a, b)
                if utility[0] > v:
                    v = utility[0]
                    chosenAction = action
                if v > b:
                    return [v, chosenAction]
                a = max(a, v)


            return [v, chosenAction]

        a = float('-inf')
        b = float('inf')
        m = maxi(gameState, 1, a, b)
        utility, action = m[0], m[1]
        return action

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
        # For the Ghosts
        def mini(state, ghost, depth):
            """
            :param state - a given state of the game
            :param ghost - the agent index of the ghost maxi is being calculated
                            for
            :param depth - the current depth calculated
            :return list - a list with two elements, the utility of that state,
                            and the action taken to get there
            """
            # moves for ghost
            legalMoves = state.getLegalActions(ghost)

            if state.isLose() or not legalMoves or self.depth < depth:
                return [self.evaluationFunction(state), None]

            # if there are still ghosts to calculate
            utilities = []
            total_ghosts = state.getNumAgents() - 1
            for action in legalMoves:
                if ghost != total_ghosts:
                    suc = state.generateSuccessor(ghost, action)
                    utilities.append(mini(suc, ghost + 1, depth))
                else:
                    suc = state.generateSuccessor(ghost, action)
                    utilities.append(maxi(suc, depth + 1))

            sum_of_expectations = 0
            for item in utilities:
                utility = item[0]
                sum_of_expectations += float(utility) / len(utilities)
            return sum_of_expectations, None

        # Pacman
        def maxi(state, depth):
            """
            :param state - a given state of the game
            :param depth - the current depth calculated
            :return list - a list with two elements, the utility of that state,
                            and the action taken to get there
            """
            # moves for pacman
            legalMoves = state.getLegalActions(0)
            if  state.isWin() or not legalMoves or self.depth < depth:
                return [self.evaluationFunction(state), None]

            utilities = []
            # get successors of actions
            for action in legalMoves:
                suc = state.generateSuccessor(0, action)
                utilities.append( (mini(suc, 1, depth)[0], action) )

            return max(utilities)

        m = maxi(gameState, 1)
        utility, action = m[0], m[1]
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
