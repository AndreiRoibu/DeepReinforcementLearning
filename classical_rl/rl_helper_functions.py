# Reinforcement Learning Helper Functions
# ------------------------------
# Code written by Andrei Roibu
# Created: 17/01/2019
# Last Modified: 17/01/2019
# ------------------------------
# This script contains a series of helper functions required for the
# SARSA and Q-learning RL algorithms
# ------------------------------

# We import the relevant packages

import numpy as np

def print_values (values,grid):

    # This functions draws the grid and prints the values
    # Arguments:
    #   V = dictoniary containing the value function
    #   grid = object containing the defined grid

    for i in range(grid.height):
        print("----------------------------------------------------------------")
        for j in range(grid.length):
            value = values.get((i,j),0)
            if value >= 0:
                print (" %.2f|" % value, end="")
            else:
                print ("%.2f|" % value, end="") # differite to allow (-) sign
        print("")

def print_policy(policy,grid):
    
    # This function draws the grid and prints the policy
    # Arguments:
    #   policy = dictionary containing the policy function
    #   grid = object cotaining the grid

    for i in range(grid.height):
        print("------------------------------------------------")
        for j in range (grid.length):
            action = policy.get((i,j),' ')
            print(" %s |" % action, end="")
        print("")

def random_action (action, epsilon = 0.1):

    # This function performs an epsilon-soft random action selection.
    # This function can be modified for epsilon-gready, by exploiting
    # the best available action.

    all_possible_actions = ('U', 'D', 'L', 'R')
    probability = np.random.random()
    if probability < (1 - epsilon):
        return action
    else:
        return np.random.choice(all_possible_actions)

def argmax_max_dict (dictionary):
    
    # This function returns the argmax (key) and max (value) of a dictionary.

    max_key = None
    max_value = float('-inf')

    for key,value in dictionary.items():
        if value > max_value:
            max_key = key
            max_value = value

    return max_key,max_value

