
"""
COMPSCI 182
Sameer Mehra and Raul Jordan
HW4
"""
import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0


        self.actions = {}
        # Creates a dictionary of actions that correspond to a given state,
        # Which will be useful later on
        for s in mdp.getStates():
            self.actions[s] = [None] * len(mdp.getStates())

        # Write value iteration code here
        for n in range(self.iterations):
            tmp = util.Counter()
            for state in self.mdp.getStates():
                maxq = float('-inf')

                for action in mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, action)

                    # If the q is bigger than the max q we calculated
                    if q > maxq:
                        maxq = q
                        tmp[state] = maxq
                        self.actions[state] = action
            self.values = tmp

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # Initializes qtotal as a counter from 0
        qtotal = 0
        probabilities = self.mdp.getTransitionStatesAndProbs(state, action)

        for s_prime, probability in probabilities:
            s_prime_reward = self.mdp.getReward(state, action, s_prime)

            # We simply add to the qtotal counter
            qtotal += probability * (s_prime_reward + self.discount * self.values[s_prime])
        return qtotal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Simply returns the action at that state in the dictionary we created
        return self.actions[state]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
