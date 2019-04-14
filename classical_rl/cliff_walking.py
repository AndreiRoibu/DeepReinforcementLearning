# Cliff Walking
# ------------------------------
# Code written by Andrei Roibu
# Created: 17/01/2019
# Last Modified: 17/01/2019
# ------------------------------
# This is a modified version of the GridWorld example, designed to
# compare SARSA and Q-learning, highlighting the differences between
# the two methods.The example is inspired from Example 6.6 in the 
# Reinforcement Learning book by Sutton and Barto (2018),p.158
# ------------------------------

class CliffWalking:
    def __init__ (self, height, length, start):

        # This represents the constructor function.
        # Inputs: dimensions and starting position

        self.height = height
        self.length = length
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):

        # This functions sets all the possible actions and the
        # rewards associated with each state.
        # Inputs: dictonaries containing a list of actions and rewards

        self.actions = actions
        self.rewards = rewards 

    def set_state(self,s):

        # This function sets the state of the agent

        self.i = s[0]
        self.j = s[1]

    def current_state(self):

        # This functions returns the current (i,j) agent state

        return (self.i, self.j)

    def all_states(self):

        # This function returns a set of all states which allow
        # actions or hold rewards.

        return set(self.actions.keys()) | set(self.rewards.keys())

    def move(self,action):

        # This function checks if an action is possible, and if it is
        # it performs it, changing the state and producing a reward.

        if action in self.actions[(self.i,self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'L':
                self.j -= 1
            elif action == 'R':
                self.j += 1
            
        return self.rewards.get((self.i,self.j), 0)

    def is_terminal(self,s):

        return s not in self.actions

    def game_over(self):

        # Returns true if the game is over = no more actions possible.

        return (self.i,self.j) not in self.actions

def game_grid(step_cost = -1.0):

    # This function constructrs the environment, and the lists of
    # actions and rewards

    grid = CliffWalking(4,12,(3,0))

    # We first defined the rewards

    # The penalty incentivises the agent to find an optimum route

    rewards = {
        (0,0): step_cost,
        (1,0): step_cost,
        (2,0): step_cost,
        (3,0): step_cost,
        (0,11): step_cost,
        (1,11): step_cost,
        (2,11): step_cost,
        (3,11): 10.0,
    }

    for j in range (1,11):
        rewards[(3,j)]=-100.0 # define the penalty for falling in the cliffe

    # We now define the actions

    actions = {
        (0,0): ('D', 'R'),
        (1,0): ('U', 'D', 'R'),
        (2,0): ('U', 'D', 'R'),
        (3,0): ('U', 'R'),
        (0,11): ('D', 'L'),
        (1,11): ('U', 'D', 'L'),
        (2,11): ('U', 'D', 'L'),
    }

    for i in range(3):
        for j in range (1,11):
            rewards[(i,j)] = step_cost
            if i == 0:
                actions[(i,j)] = ('D', 'L', 'R')
            else:
                actions[(i,j)] = ('U', 'D', 'L', 'R')

    grid.set(rewards,actions)
            
    return grid