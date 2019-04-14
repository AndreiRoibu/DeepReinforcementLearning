# DSN for CartPole (version 1.0)
# ------------------------------
# Code written by Andrei Roibu
# Created: 13/02/2019
# Last Modified: 01/03/2019
# ------------------------------
# This script looks at implementing the DSN algorithm for solving
# control problem in the game of CartPole.
# The code used in this script uses the Tensorflow framework. This is
# due to the fact that the previous framework, Theano, did not allow
# the calculations of gradients between two tensors, required for the
# DQN-DFA and DSN-DFA codes.
# This piece of code is designed to serve as the starting point for
# the DSN-DFA codes.
# The first version of this code uses both the ADAM optimiser and the 
# RMSprop optimisers resented in the DQN paper. This if for
# simplicity.
# Much of the base code was inspired by the "Advanced AI: Deep 
# Reinforcement Learning in Python" course on udemy, available at
# https://www.udemy.com/deep-reinforcement-learning-in-python/
# ------------------------------

# We start by importing the relevant packages

import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

from gym import wrappers
from datetime import datetime

# We start by defining a function for plotting the running average.
# This will be usefull in assessing the performance of the algorithm.

def plot_running_average(total_rewards):
    N = len(total_rewards)
    running_average = np.empty(N)
    for i in range(N):
        running_average[i] = total_rewards[max(0,i-100):(i+1)].mean()

    return running_average

# As opposed to the Theano-based code, we do not need to define the 
# Adam and RMSprop optimizers, as they are already defined within
# Teonsorflow.

# We now start coding the HiddenLayer class.
# This initializes the hidden layer parameters, based on the input
# and output sizes, and also defines the activation function. 
# For now, we will use the Relu activation function, but this can be
# changed later. 

class HiddenLayer:
    def __init__ (self, M1, M2, activation_function = tf.nn.relu, use_bias = True):
        self.W = tf.Variable(tf.truncated_normal(shape= (M1,M2)) * np.sqrt(2/(M1+M2)))
        self.parameters = [self.W]
        
        self.use_bias = use_bias

        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.parameters.append(self.b)
        
        self.activation_function = activation_function

    def forward(self,X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.activation_function(a)

# Now, we define the DSN class.

class DSN:
    def __init__ (self, inputs, outputs, hidden_layer_sizes, gamma, learning_rate, decay, eps, max_experiences= 10000, min_train_experiences= 100, batch_size= 32):

        # The constructor thakes the following inputs:
        #   - inputs: number of inputs
        #   - outputs: number of output actions
        #   - hidden_layer_sizes: the size of the hidden layers
        #   - gamma: discout factor
        #   - max_experiences: the size of the experience reply buffer
        #   - min_traiin_experiences: the minimum number of experiences to be collected before training
        #   - batch_size: the number of samples used for training

        # We then start creating the NN layers.

        self.outputs = outputs

        self.layers = []
        M1 = inputs
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1 = M2

        final_layer = HiddenLayer(M1, outputs, lambda x:x)
        self.layers.append(final_layer)

        # After this has been done, we collect all the parameters in
        # the newtork as this will be necessary when creating the 
        # seconday network. This will also help in updating the
        # parameters.

        self.parameters = []
        for layer in self.layers:
            self.parameters += layer.parameters

        # Having done this, we declare the input and the target
        # tensorflow variables.

        self.X = tf.placeholder(tf.float32, shape= (None, inputs), name= 'X')
        self.G = tf.placeholder(tf.float32, shape= (None,), name= 'G')
        self.actions = tf.placeholder(tf.int32, shape= (None,), name= 'actions')

        # We also calculate the outputs and the cost

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z

        # We set the prediction operation to be Y-hat, as this is
        # the desired model output. 

        self.prediction_operation = Y_hat

        # Then, we select only the values that correspond to actions
        # that we actually took. Then, we calculate the cost.

        # Only the entries of Y-hat that correspond to actions that
        # we actually took matter. Thus, we either have to index Y-hat
        # or we can create a one-hot matrix. This allows us to get
        # rid of the O's. The Theano approach seems a little more 
        # straight forward. 

        selected_action_values = tf.reduce_sum(
            Y_hat * tf.one_hot(self.actions, outputs),
            reduction_indices= [1]
        )

        # After this, we calculate the cost and the updates. 

        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        
        self.training_operation = tf.train.RMSPropOptimizer(learning_rate= learning_rate, decay= decay, epsilon= eps).minimize(cost)

        # Next, we create the replay memory.

        self.experience = {
            's': [],
            'a': [],
            'r': [],
            's2' : [],
            'done': []
        }

        self.max_experiences = max_experiences
        self.min_train_experiences = min_train_experiences
        self.batch_size = batch_size
        self.gamma = gamma

        # This marks the end of the constructor. Now we build a few
        # additonal functions

    # First, we create the set_session function. This is a basic setter.
    
    def set_session(self, session):
        self.session = session

    def copy_parameters(self, other):

        # This function copies all the parameters from the input
        # network. All the values in the input netork are copied to
        # the corresponding parameters in the other network.

        # These are all operations, so we need to run them inside a
        # session

        operations = []
        original_parameters = self.parameters
        copied_parameters = other.parameters

        for param1, param2 in zip(original_parameters, copied_parameters):
            actual_value = self.session.run(param2)
            operation = param1.assign(actual_value)
            operations.append(operation)

        # We finish by running all of them

        self.session.run(operations)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.prediction_operation, feed_dict = {self.X: X})

    def train(self, target_network, epsilon):

        # This function trains the NN
        # If we have not collected enough experience, the function
        # returns without doing anything.

        if len(self.experience['s']) < self.min_train_experiences:
            return
        
        # Otherwise, we select a set of random indexes equal to the 
        # batch size. We then use these indexes to get the states, 
        # actions, rewards and next states.

        indexes = np.random.choice(len(self.experience['s']), size= self.batch_size, replace= False)

        states = [self.experience['s'][index] for index in indexes]
        actions = [self.experience['a'][index] for index in indexes]
        rewards = [self.experience['r'][index] for index in indexes]
        next_states = [self.experience['s2'][index] for index in indexes]
        dones = [self.experience['done'][index] for index in indexes]

        # The next states are then used to predict the Q values.
        # Then, the current epsilon is used to selec the following
        # actions and their corresponding Q-values for training.

        Q_values = target_network.predict(next_states)

        Q_predicted = []

        for index in range(Q_values.shape[0]):
            X = Q_values[index]
            exps = np.exp(X - np.max(X))
            
            if exps.ndim==1:
                distribution = exps / np.sum(exps, axis=0)
            else:
                distribution = exps /np.sum(exps, axis=1, keepdims=True)

            cumulative_distribution=np.empty(len(distribution)+1)

            for i in range(len(distribution)+1):
                if i == 0:
                    cumulative_distribution[i]=0.0
                else:
                    cumulative_distribution[i] = cumulative_distribution[i-1] + distribution[i-1]
            
            probability = np.random.random()
            while probability >= cumulative_distribution[-1]:
                probability = np.random.random()

            location = np.max(np.where(cumulative_distribution<probability))
            Qs = Q_values[index][location]
            Q_predicted.append(Qs)

        Q_predicted = np.atleast_1d(Q_predicted)

        # Then, using these, we can calculate the targets, or returns (G)

        targets = [reward + self.gamma * next_Q if not done else reward for reward, next_Q, done in zip(rewards, Q_predicted, dones)]

        # Finally, using this data, we call the training optimizer

        self.session.run(
            self.training_operation,
            feed_dict= {
                self.X: states,
                self.G: targets,
                self.actions: actions
            }
        )

    def add_experience(self, state, action, reward, next_state, done):

        # This function ads the 4-touple to the experience reply list.
        # If we have reached the maximum number of elements, we 
        # remove the first element and then append the new one to the
        # end.

        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)

        self.experience['s'].append(state)
        self.experience['a'].append(action)
        self.experience['r'].append(reward)
        self.experience['s2'].append(next_state)
        self.experience['done'].append(done)

    def sample_action(self, x, epsilon):

        # This function will be requred for sampling the actions.
        # This version of the function uses a softmax approach.

        X = self.predict(np.atleast_2d(x))[0]

        exps = np.exp(X - np.max(X))
        if exps.ndim==1:
            distribution = exps / np.sum(exps, axis=0)
        else:
            distribution = exps /np.sum(exps, axis=1, keepdims=True)

        cumulative_distribution=np.empty(len(distribution)+1)

        for i in range(len(distribution)+1):
            if i == 0:
                cumulative_distribution[i]=0.0
            else:
                cumulative_distribution[i] = cumulative_distribution[i-1] + distribution[i-1]

        probability = np.random.random()
        while probability >= cumulative_distribution[-1]:
            probability = np.random.random()

        return np.max(np.where(cumulative_distribution<probability))


# Having finished defining our model, we define a function which plays
# one episode of the game.

def play_one_episode(environmnet, network, target_network, epsilon, gamma, copy_period, total_steps, penalty):

    # The input copy_period informs us how often to copy our main 
    # network to the target network

    observation = environmnet.reset()
    done = False
    total_reward = 0
    iterations = 0

    action_1 = network.sample_action(observation, epsilon)
    action_t = action_1

    while not done and iterations < 2000:
        
        if total_steps % copy_period == 0:
            target_network.copy_parameters(network)

        previous_observation = observation
        observation, reward, done, _ = environmnet.step(action_t)

        total_reward += reward

        if done:
            reward = penalty # We penalize the agent if the game finishes early
    
        # After taking an action, we store the data in our experience
        # replay database, and then we update the model.

        network.add_experience(previous_observation, action_t, reward, observation, done)
        network.train(target_network,epsilon)

        action_t = network.sample_action(observation, epsilon)

        total_steps += 1

    return total_reward, total_steps

def play_evaluation_episode(environmnet, network, epsilon, penalty):

    observation = environmnet.reset()
    done = False
    total_evaluation_reward = 0
    iterations = 0

    action_1 = network.sample_action(observation, epsilon)
    action_t = action_1

    while not done and iterations < 2000:
        
        observation, reward, done, _ = environmnet.step(action_t)

        total_evaluation_reward += reward

        if done:
            reward = penalty # We penalize the agent if the game finishes early

        action_t = network.sample_action(observation, epsilon)

    return total_evaluation_reward

def main(console_file, learning_rate= 1e-4, decay= 0.99, gamma = 0.99, eps= 1e-3, copy_period=200, episodes=1000, penalty=0):
    environment = gym.make('CartPole-v0')
    environment._max_episode_steps = 500

    evaluation_period = 200
    evaluation_episodes = 100
    final_evalution_episodes = 1000
    
    inputs = len(environment.observation_space.sample())
    outputs = environment.action_space.n
    hidden_layer_sizes = [200,200]

    network = DSN(inputs, outputs, hidden_layer_sizes, gamma, learning_rate, decay, eps)
    target_network = DSN(inputs, outputs, hidden_layer_sizes, gamma, learning_rate, decay, eps)

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    network.set_session(session)
    target_network.set_session(session)

    total_rewards = np.empty(episodes)
    mean_evaluation_rewards = []
    standard_deviation_rewards = []

    total_steps = 0

    for episode in range(episodes):
        epsilon = 1.0 / np.sqrt(episode + 1)
        total_reward, total_steps = play_one_episode(environment, network, target_network, epsilon, gamma, copy_period, total_steps, penalty)
        total_rewards[episode] = total_reward

        if episode % 100 == 0:
            print("Episode:", episode, "Total Reward:", total_reward, "Epsilon:", epsilon, "Average Reward (last 100 episodes):", total_rewards[max(0, episode - 100):(episode + 1)].mean())
            print("Episode:", episode, "Total Reward:", total_reward, "Epsilon:", epsilon, "Average Reward (last 100 episodes):", total_rewards[max(0, episode - 100):(episode + 1)].mean(), file=open(console_file,'a'))
            
        if episode % evaluation_period == 0 and episode != 0:

            evaluation_rewards = np.empty(evaluation_episodes)

            for evaluation_episode in range(evaluation_episodes):
                
                evaluation_reward = play_evaluation_episode(environment, network, epsilon, penalty)
                evaluation_rewards[evaluation_episode] = evaluation_reward
            
            mean_evaluation_reward = np.mean(evaluation_rewards)
            standard_deviation_reward = np.std(evaluation_rewards)

            mean_evaluation_rewards.append(mean_evaluation_reward)
            standard_deviation_rewards.append(standard_deviation_reward)

        if episode == episodes-1:

            evaluation_rewards = np.empty(evaluation_episodes)

            for evaluation_episode in range(evaluation_episodes):
                
                evaluation_reward = play_evaluation_episode(environment, network, epsilon, penalty)
                evaluation_rewards[evaluation_episode] = evaluation_reward

            mean_evaluation_reward = np.mean(evaluation_rewards)
            standard_deviation_reward = np.std(evaluation_rewards)

            mean_evaluation_rewards.append(mean_evaluation_reward)
            standard_deviation_rewards.append(standard_deviation_reward)

    final_rewards = np.empty(final_evalution_episodes)

    for evaluation_episode in range(final_evalution_episodes):
                
        evaluation_reward = play_evaluation_episode(environment, network, epsilon, penalty)
        final_rewards[evaluation_episode] = evaluation_reward

    mean_final_reward = np.mean(final_rewards)
    standard_deviation_final_reward = np.std(final_rewards)

    print("Average Reward after the last 100 episodes:", total_rewards[-100:].mean())
    print("Total Steps:", total_rewards.sum())

    print("The average final evalution reward:", mean_final_reward)
    print("The standard deviation of the final evalution reward:", standard_deviation_final_reward)

    print("Average Reward after the last 100 episodes:", total_rewards[-100:].mean(), file=open(console_file,'a'))
    print("Total Steps:", total_rewards.sum(),file= open(console_file,'a'))

    print("The average final evalution reward:", mean_final_reward, file=open(console_file, 'a'))
    print("The standard deviation of the final evalution reward:", standard_deviation_final_reward, file=open(console_file,'a'))

    running_average_reward = plot_running_average(total_rewards)

    session.close()   

    return total_rewards, mean_evaluation_rewards, standard_deviation_rewards, mean_final_reward, standard_deviation_final_reward, running_average_reward, final_rewards


if __name__ == '__main__':

    idf = 'BK'
    ida = '_ADF'
    console_file = idf+ida+'.txt'

    # penaltys = [0, -100, -200, -300, -400, -500]
    penaltys = [0]
    # training_episodes = [500,1000,2000,3000]
    # learning_rates = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
    # decays = [0.99, 0.75, 0.5, 0.25, 0.1]
    # gammas = [0.99, 0.75, 0.5, 0.25, 0.1]
    # epsilons = [1e-1, 1e-3, 1e-5, 1e-8]
    # copy_periods = [1000, 500, 200, 100, 50]

    counter = penaltys

    total_rewards_out = []
    mean_evaluation_rewards_out = [] 
    standard_deviation_rewards_out = []
    mean_final_reward_out = []
    standard_deviation_final_reward_out = []
    running_average_reward_out = []

    for iterator in range(len(counter)):
        total_rewards, mean_evaluation_rewards, standard_deviation_rewards, mean_final_reward, standard_deviation_final_reward, running_average_reward, final_rewards = main(
            console_file, penalty= penaltys[iterator])
        
        total_rewards_out.append(total_rewards)
        mean_evaluation_rewards_out.append(mean_evaluation_rewards)
        standard_deviation_rewards_out.append(standard_deviation_rewards)
        mean_final_reward_out.append(mean_final_reward)
        standard_deviation_final_reward_out.append(standard_deviation_final_reward)
        running_average_reward_out.append(running_average_reward)

    with open('rewards'+idf+ida+'.csv', "a") as output_file:
        writer = csv.writer(output_file)
        for i in range(len(counter)):
            writer.writerow(total_rewards_out[i])

    with open('eval'+idf+ida+'.csv', "a") as output_file:
        writer = csv.writer(output_file)        
        for i in range(len(counter)):
            writer.writerow(mean_evaluation_rewards_out[i])

    with open('std_eval'+idf+ida+'.csv', "a") as output_file:
        writer = csv.writer(output_file)
        for i in range(len(counter)):
            writer.writerow(standard_deviation_rewards_out[i])

    with open('final'+idf+ida+'.csv', "a") as output_file:
        writer = csv.writer(output_file)   
        for i in range(len(counter)):
            writer.writerow([mean_final_reward_out[i]])

    with open('std_final'+idf+ida+'.csv', "a") as output_file:
        writer = csv.writer(output_file)   
        for i in range(len(counter)):
            writer.writerow([standard_deviation_final_reward_out[i]])

    with open('run_avg'+idf+ida+'.csv', "a") as output_file:
        writer = csv.writer(output_file)   
        for i in range(len(counter)):
            writer.writerow(running_average_reward_out[i])

    with open('final_rewards'+idf+ida+'.csv', "a") as output_file:
        writer = csv.writer(output_file)
        for i in range(len(counter)):
            writer.writerow(final_rewards)
