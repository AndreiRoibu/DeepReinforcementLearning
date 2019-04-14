# Q-learning
# ------------------------------
# Code written by Andrei Roibu
# Created: 17/01/2019
# Last Modified: 17/01/2019
# Code after: https://lazyprogrammer.me
# ------------------------------
# This piece of code looks at solving the control problem in Gridworld by using the Q-learning method.
# In this piece of code we will aim to optimise the given policy.
# ------------------------------

# We start by importing the relevant packages
import csv
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from rl_helper_functions.py import print_policy, print_values, argmax_max_dict

# Declare the Hyper Parameters

gamma = 0.9
alpha = 0.1
# t for updating epsilon
all_possible_actions =  ('U', 'D', 'L', 'R')

def random_action (a,eps=0.1):
    p = np.random.random()
    if p < (1-eps):
        return a
    else:
        return np.random.choice(all_possible_actions)

# We then start the code. We do not have a playgame function, as playing the game and doing the updates cannot be separate
# The updates need to be done while playing the game

if __name__ == '__main__' :

    grid = negative_grid(step_cost = -0.1)

    # we initialize Q(s,a)
    # then, we track the updates of Q(s,a)

    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s]={}
        for a in all_possible_actions:
            Q[s][a] = 0.0

    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in all_possible_actions:
            update_counts_sa[s][a] = 1.0

    # now we start playing the game

    t = 1.0
    deltas = []
    rewards = []
    for it in range (10000):
        if it % 100 == 0:
            t += 1e-2
        if it % 1000 == 0:
            print("it:",it,"/10000")

        s = (2,0) # the start position
        grid.set_state(s)

        a = max_dict(Q[s])[0]
        delta = 0
        reward= 0
        while not grid.game_over():
            a = random_action(a,eps = 0.5/t)
            r = grid.move(a)
            s2 = grid.current_state()

            alpha_game = alpha / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005
            
            old_Qsa = Q[s][a]

            a2, maxQs2a2 = max_dict(Q[s2])
            Q[s][a] = Q[s][a] + alpha_game * (r + gamma*maxQs2a2 - Q[s][a])

            delta = max(delta,np.abs(old_Qsa - Q[s][a]))

            update_counts[s] =update_counts.get(s,0)+1

            s = s2
            a = a2

            reward += r

            
        rewards.append(reward)

        deltas.append(delta)
    
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

    # determine the policy, and find optimum value functions

    policy = {}
    V = {}
    for s in grid.actions.keys():
        a,maxQ = max_dict(Q[s])
        policy[s]  = a
        V[s] = maxQ

    # what's the proportion of time we spend updating each part of Q?
    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)

    with open('maze_q_rewards.csv', "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(running_average)

    with open('maze_q_deltas.csv', "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(running_average_delta)