# SARSA
# ------------------------------
# Code written by Andrei Roibu
# Created: 17/01/2019
# Last Modified: 17/01/2019
# Code after: https://lazyprogrammer.me
# ------------------------------
# This piece of code looks at solving the control problem in Gridworld by using the SARSA method.
# In this piece of code we will aim to optimise the given policy.
# ------------------------------

# We start by importing the relevant packages
import csv
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid,negative_grid
from iterative_policy_evaluation import print_policy,print_values
from monte_carlo_es import max_dict

# We then declare the hyperparameters

gamma = 0.9
alpha = 0.1 # not the actual alpha, alpha will be derived from this initial alpha
all_possible_actions = ('U', 'D', 'L', 'R')
# t (for updating epsilon) is another hyper parameter

def random_action(a, eps=0.1):
    p = np.random.random() # returns a random float from [0,1)
    if p < (1-eps):
        return a
    else:
        return np.random.choice(all_possible_actions)

# We then start the code. We do not have a playgame function, as playing the game and doing the updates cannot be separate
# The updates need to be done while playing the game

if __name__ == '__main__':

    grid = negative_grid(step_cost=-0.1) #we want to penalize each agent movement

    # We then initialize Q(s,a)
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s]={} # we initialize as dictionary, as this also acts as policy
        for a in all_possible_actions:
            Q[s][a]=0
    
    # We also keep track of how oftern Q[s] has been updated.
    # This is needed for updating the learning rate
    
    update_counts={}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in all_possible_actions:
            update_counts_sa[s][a] = 1
    
    t = 1.0 # time value used for updating epsilon
    deltas = []
    rewards = []
    for it in range (10000): # we start playing the game
        if it % 100 == 0:
            t += 1e-2 # how often and by how much we increae t represents a HYPERPARAMETER
        if it % 1000 == 0:
            print("it:",it,"/10000")

        # instead of generating an epsisod, we will play an epsisode withing the loop

        s = (2,0) # starting position
        grid.set_state(s)

        # the first (s,r) tuple is the start state and is equal to 0 (no rewards for starting the game)
        # the final (s,r) is the terminal state = 0 (by definition) -> no need to update it

        a = max_dict(Q[s])[0]
        a = random_action(a,eps = 0.5/t) # epsilon greedy, as the action is choosen between the best available action and a random exploration action

        delta = 0
        reward = 0

        while not grid.game_over():

            r = grid.move(a)
            s2 = grid.current_state()

            a2 = max_dict(Q[s2])[0] # we need the next action for Q(s',a')
            a2 = random_action(a2,eps = 0.5/t) 

            # We update Q(s,a) as we experience the episode

            alpha_game = alpha / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005 # only update the count by a small amount

            old_Qsa = Q[s][a]

            Q[s][a] = Q[s][a] + alpha_game * (r + gamma * Q[s2][a2] - Q[s][a])

            delta = max(delta, np.abs(old_Qsa - Q[s][a]))

            update_counts[s] = update_counts.get(s,0) + 1 # we keep track of how often Q(s) has been updated

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

    # now, we can determine the the optimal policy from Q*
    # we can also find the optimal value function V* from Q*

    policy = {}
    V = {}
    for s in grid.actions.keys():
        a,max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # finally, we look at the proportion of time spent updating each part of Q

    print("\nUpdate counts: ")
    total = np.sum(list(update_counts.values()))
    for k,v in update_counts.items():
        update_counts[k] = float(v)/total
    print_values(update_counts,grid)

    print("\nvalues:")
    print_values(V,grid)
    print("\npolicy:")
    print_policy(policy,grid)

    with open('maze_sarsa_rewards.csv', "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(running_average)

    with open('maze_sarsa_deltas.csv', "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(running_average_delta)