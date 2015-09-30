# search.py
# ---------------
# Raul E. Jordan
# Harvard Class of 2017
# CS182

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    start_node = (problem.getStartState(), [], 0)
    explored = []
    frontier = util.Stack()
    frontier.push(start_node)

    if problem.isGoalState(problem.getStartState()):
        return []

    while not frontier.isEmpty():
        (current_state, actions, costs) = frontier.pop()
        if current_state not in explored:
            explored.append(current_state)
            if problem.isGoalState(current_state):
                return actions
            else:
                return []
            for child_node in problem.getSuccessors(current_state):
                next_state = child_node[0]
                next_action = child_node[1]
                next_cost = child_node[2]

                next_node = (next_state, actions + [next_action], costs + next_cost)
                frontier.push(next_node)
    return []


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    start_node = (problem.getStartState(), [], 0)
    explored = []
    frontier = util.Queue()
    frontier.push(start_node)

    if problem.isGoalState(problem.getStartState()):
        return []

    while not frontier.isEmpty():
        (current_state, actions, costs) = frontier.pop()
        if current_state not in explored:
            explored.append(current_state)
            if problem.isGoalState(current_state):
                return actions
            for child_node in problem.getSuccessors(current_state):
                next_state = child_node[0]
                next_action = child_node[1]
                next_cost = child_node[2]

                next_node = (next_state, actions + [next_action], costs + next_cost)
                frontier.push(next_node)
    return []


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    start_node = (problem.getStartState(), [], 0)
    explored = []
    frontier = util.PriorityQueue()
    start_cost = 0
    frontier.push(start_node, start_cost)

    if problem.isGoalState(problem.getStartState()):
        return []

    while not frontier.isEmpty():
        (current_state, actions, costs) = frontier.pop()
        if current_state not in explored:
            explored.append(current_state)
            if problem.isGoalState(current_state):
                return actions
            for child_node in problem.getSuccessors(current_state):
                next_state = child_node[0]
                next_action = child_node[1]
                next_cost = child_node[2]

                g = problem.getCostOfActions(actions + [next_action])

                next_node = (next_state, actions + [next_action], g)
                frontier.push(next_node, g)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."

    start_node = (problem.getStartState(), [], 0)
    explored = []
    frontier = util.PriorityQueue()
    start_cost = heuristic(problem.getStartState(), problem)
    frontier.push(start_node, start_cost)

    if problem.isGoalState(problem.getStartState()):
        return []

    while not frontier.isEmpty():
        (current_state, actions, costs) = frontier.pop()
        if current_state not in explored:
            explored.append(current_state)
            if problem.isGoalState(current_state):
                return actions
            for child_node in problem.getSuccessors(current_state):
                next_state = child_node[0]
                next_action = child_node[1]
                next_cost = child_node[2]

                g = problem.getCostOfActions(actions + [next_action])
                h = heuristic(next_state, problem)
                g_h = g + h

                next_node = (next_state, actions + [next_action], costs + next_cost)
                frontier.push(next_node, g_h)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
