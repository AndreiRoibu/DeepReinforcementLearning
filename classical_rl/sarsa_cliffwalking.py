# SARSA
# ------------------------------
# Code written by Andrei Roibu
# Created: 17/01/2019
# Last Modified: 17/01/2019
# ------------------------------
# This script looks at solving the control problem in GridWorld 
# Cliff Walking by using the SARSA method.
# In this piece of code we will aim to optimise the given policy.
# ------------------------------

# We start by importing the relevant packages
import csv
import numpy as np
import matplotlib.pyplot as plt
from cliff_walking import game_grid
from rl_helper_functions import print_values, print_policy, random_action, argmax_max_dict

# We then declare the hyperparameters

gamma = 0.9 # discount factor
alpha_zero = 0.1 # original learning rate

all_possible_actions = ('U', 'D', 'L', 'R')

if __name__ == '__main__':

    grid = game_grid(step_cost = -0.1)

    t = 1.0 # time value required for updating epsilon

    # We initialize Q(s,a) and the states
    Q = {}
    sa_count = {} # necessary for updating the learning rate
    states = grid.all_states()
    for s in states:
        Q[s]={}
        sa_count[s]={}
        for a in all_possible_actions:
            Q[s][a] = 0
            sa_count[s][a] = 1

    deltas = []
    rewards = []

    # Start the episodic training process

    for episode in range (10000):
        if episode % 100 == 0:
            t += 1e-2
        if episode % 1000 == 0:
            print("Played",episode,"episodes out of 10,000")

        # Define the starting position

        s = (3,0)
        grid.set_state(s)

        a = argmax_max_dict(Q[s])[0]
        a = random_action(a,epsilon = 0.5/t)

        delta = 0

        # We play the game

        reward = 0

        while not grid.game_over():

            r = grid.move(a)
            s2 = grid.current_state()

            a2 = argmax_max_dict(Q[s2])[0]
            a2 = random_action(a2,epsilon = 0.5/t)

            alpha = alpha_zero / sa_count[s][a]
            sa_count[s][a] += 0.005

            Q_old = Q[s][a]
            Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s2][a2] - Q[s][a])

            delta = max(delta, np.abs(Q_old - Q[s][a]))

            s = s2
            a = a2

            reward += r
        
        deltas.append(delta)
        rewards.append(reward)

    # This completes the training   
    # Now we need to visualise the policy and determine the optimal V*,Q*

    policy = {}
    V = {}
    for s in grid.actions.keys():
        a,Q_max = argmax_max_dict(Q[s])
        policy[s] = a
        V[s] = Q_max

    print("\nvalues:")
    print_values(V,grid)
    print("\npolicy:")
    print_policy(policy,grid)
    print("\ndeltas:")

    deltas2 = np.asarray(deltas)

    Nd = len(deltas2)
    running_average_delta = np.empty(Nd)
    for i in range(Nd):
        running_average_delta[i] = deltas2[max(0,i-2000):(i+1)].mean()

    plt.plot(running_average_delta)
    plt.show()

    rewards2 = np.asarray(rewards)

    N = len(rewards2)
    running_average = np.empty(N)
    for i in range(N):
        running_average[i] = rewards2[max(0,i-2000):(i+1)].mean()

    plt.plot(running_average)
    plt.show()

    with open('cliff_sarsa_rewards.csv', "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(running_average)

    with open('cliff_sarsa_deltas.csv', "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(running_average_delta)