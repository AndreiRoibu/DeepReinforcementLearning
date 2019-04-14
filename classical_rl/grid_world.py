# GridWorld
# ------------------------------
# Code written by Andrei Roibu
# Created: 15/01/2019
# Last Modified: 15/01/2019
# Code after: https://lazyprogrammer.me
# ------------------------------
# This piece of code is destined to createa the Environment
# for a set of subsequent simple pieces of Reinforcement Learning
# codes. In this code, we will first define the environments itsef
# and after that define the standard and negative grids. These are
# required in order to provide the rewards for the learning agent.

# We start by loading the required packages

import numpy as np

# We now define our Grid class, which represents the Environment.
#
# We first define the class, which is an object allowing us to
# bundle data and funcitonality together. In our case, the environment.
# In other words, te class is a set or category of things having some
# property or attribute in common and differentiated forms.
#
# The object is one of the instances of the class, which can perform
# the functionalities which are defined in the class.
#
# The self represents the instances of the class, which allows us to
# access the attributes and methods of the class.
#
# Finally, __init__ is a special kind of class known as the constructor.
# This is used for creating the objct from the class and it allows the
# class to inialize the atributes of a class. 

class GridWorld:
    def __init__ (self,width,height,start): # constructor takes in dimensions
        self.width = width 
        self.height = height
        self.i = start[0] #this defines the start position = a touple of 2 integers
        self.j = start[1] # i and j store the current instance location

    def set(self,rewards,actions):

        # This function sets both the rewards and actions of the
        # environment simultaneously. 
        # Actions include all possible actions that can take you to a new state. 
        # Rewards include all rewards associated with a given state.
        # Actions should be a dictionary of form - (i,j):A(row,col) - A is a list of possible actions
        # Rewards should be a dictionary of form - (i,j):r(row,col) - r the reward

        self.rewards = rewards
        self.actions = actions

    def set_state(self,s):
        
        # This function sets the state in which we find the agent

        self.i = s[0]
        self.j = s[1]

    def current_state(self):

        # This function returns the current (i,j) position of the agent

        return(self.i,self.j)

    def is_terminal(self,s):

        # This function returns a boolean (T) if the state is a terminal state.
        # It determines this by verifying if the state is in the actions dictionary. If you can do an action in this state, than the state is not terminal.

        return s not in self.actions

    def move(self,action):

        # The function firsts checks if an action is in the actions dictionary.
        # If the action is not in the actions dictionary, than this movement cannot be done.
        # Otherwise, the agent does the action.
        # Grid conditions are used, which is why upper left corner is (0,0)
        
        # Arguments: actions - dictionary

        if action in self.actions[(self.i,self.j)]:
            if action == 'U':
                self.i -= 1 
            elif action == 'D':
                self.i += 1 
            elif action == 'R':
                self.j += 1 
            elif action == 'L':
                self.j -= 1

        # Based on the new position, the function returns a reward.

        return self.rewards.get((self.i,self.j), 0)

    def undo_move (self,action):

        # This function undoes an the action that was just performed. 

        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1

        # This acts as a sanity check, to verify if the current state is within the list of allowed states.

        assert (self.current_state() in self.all_states())

    def game_over(self):

        # This retuns true if the game is over, ie. if the agent is in a state where no more actions are possible.

        return (self.i,self.j) not in self.actions

    def all_states(self):

        # Enumerates all the states from which we can either take an action or get a rewards.

        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():

    # This function defines a grid in line with the environment requirements previously defined. 
    # It also defines the rewards for each state and a list of all possible actions. 
    # The agent cannot walk into a wall or a terminal state.

    grid = GridWorld(3,4, (2,0))

    rewards = {(0,3):1, (1,3):-1}

    actions = {
        (0,0): ('D','R'),
        (0,1): ('L','R'),
        (0,2): ('L','R','D'),
        (1,0): ('U','D'),
        (1,2): ('D','R','U'),
        (2,0): ('U','R'),
        (2,1): ('L','R'),
        (2,2): ('U','R','L'),
        (2,3): ('U','L'),
    }

    grid.set(rewards,actions)

    return grid

def negative_grid(step_cost = -0.1):

    # The negative grid penalizes each move. This is to encourage the agent to find the most optimum route and not just move around randomly.
    # The effect is that is minimizes the number of moves.

    Ngrid = standard_grid ()

    Ngrid.rewards.update({
        (0,0): step_cost,
        (0,1): step_cost,
        (0,2): step_cost,
        (1,0): step_cost,
        (1,2): step_cost,
        (2,0): step_cost,
        (2,1): step_cost,
        (2,2): step_cost,
        (2,3): step_cost,
    })

    return Ngrid