# DSN for Breakout (version 1.0)
# ------------------------------
# Code written by Andrei Roibu
# Created: 25/02/2019
# Last Modified: 25/03/2019
# ------------------------------
# This script looks at implementing the DSN algorithm for solving
# control problem in the Atari game of Breakout.
# The code used in this script uses the Tensorflow framework.
# This piece of code is designed to serve as the starting point for
# the DSN, DSN-DFA and DSN-DFA codes.
# ------------------------------

# We start by importing the relevant packages

import copy
import os
import sys
import random
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import csv

from gym import wrappers
from datetime import datetime
from scipy.misc import imresize

# Then, we define some constants that will be used throughout

max_experiences = 500000 # Total size of replay buffer 
min_train_experiences = 50000 # Minimum number of experiences before starting training
copy_period = 10000 # Training steps between copying the main network to the target network
frame_size = 84 # Size of each frame when it is imputed to the CNN

# Next, we create a class which will transform the raw input image
# into the desired input for the CNN. First, we convert the image
# to grayscale, then we resize and crop it. 

# The input image is given by the image from the game, with the
# characteristic image. 

class ImageTransformer:
    def __init__(self):
        with tf.variable_scope("image_transformer"):
            self.input_image = tf.placeholder(shape= [210, 160, 3], dtype= tf.uint8) # we store the image as uint8 8-bit integer to save space
            self.output_image = tf.image.rgb_to_grayscale(self.input_image)
            self.output_image = tf.image.crop_to_bounding_box(self.output_image, 34, 0, 160, 160)
            self.output_image = tf.image.resize_images(self.output_image, [frame_size, frame_size], method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output_image = tf.squeeze(self.output_image) # removes dimensions of size 1 from the image tensor
    
    def transform_image(self, state, session= None):
        transform_session = session or tf.get_default_session()
        return transform_session.run(self.output_image, feed_dict= {self.input_image: state})

# We now define a function which creates the new states. This function
# updates the current state by appending the new image and discarding
# the old image. This is useful in forming the new states

def update_state(state, new_image):
    return np.append(state[:,:,1:], np.expand_dims(new_image, 2), axis= 2)

# Next, we create the replay memory class. Here, we pre-allocate all
# the frames that we will store.

class Replay_Memory:
    def __init__(self, memory_size= max_experiences, frame_height= frame_size, frame_width= frame_size, frames_per_state= 4, batch_size= 32):

        # In the constructor we create relevant variables and pre-allocate
        # the required memory, for both the memory replay and the batches
        # that we will use.

        self.memory_size = memory_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frames_per_state = frames_per_state
        self.batch_size = batch_size
        self.count = 0
        self.current = 0 # Count and current keep track of the insertion point in our current buffer

        self.frames = np.empty((self.memory_size, self.frame_height, self.frame_width), dtype= np.uint8)
        self.actions = np.empty(self.memory_size, dtype= np.int32)
        self.rewards = np.empty(self.memory_size, dtype= np.float32)
        self.dones = np.empty(self.memory_size, dtype= np.bool)

        self.states = np.empty((self.batch_size, self.frames_per_state, self.frame_height, self.frame_width), dtype= np.uint8)
        self.future_states = np.empty((self.batch_size, self.frames_per_state, self.frame_height, self.frame_width), dtype= np.uint8)
        self.indices = np.empty(self.batch_size, dtype= np.int32)

    def add_experience(self, frame, action, reward, done):

        # This function adds the a touple consisting of the frame, action
        # reward, and the done. 
        # This function makes use of the current and count variables.
        # All the variables are added at their current location, which
        # is then incremented. When we reach the memory size, we loop in 
        # a circular manner, and start replacing the oldest memory with
        # the newest one.

        self.frames[self.current, ...] = frame
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.dones[self.current] = done

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size # =0 when current = memory size

    def _generate_state(self, index):

        # For a given index, this function returns the state containing
        # the previous 4 images/frames

        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.frames_per_state - 1:
            raise ValueError("Index must be at least 3!")
        return self.frames[index - self.frames_per_state + 1: index + 1, ...]

    def _generate_valid_indices(self):

        # This is a helper function for sampling a batch. 
        # This function assigns indices to be the indeces corresponding
        # to the samples in the batch and then saves them in
        # self.indices. The main purpose of this function is to check
        # for edge cases.

        for i in range(self.batch_size):
            while True:
                index = np.random.randint(self.frames_per_state, self.count - 1) # We select a random index between 4 and the size of the count variable
                if index >= self.current and index - self.frames_per_state <= self.current:
                    continue # check if the index does not cross the boundaries of self.current, which is the point where we insert a new frame
                if self.dones[index - self.frames_per_state: index].any():
                    continue # Returns True if any element in the iterable range is a terminal state
                break
            self.indices[i] = index
    
    def get_minibatch(self):

        # This function returns a minibatch of size minibatch_size

        if self.count < self.frames_per_state:
            raise ValueError("Not enough memories to get a minibatch")

        self._generate_valid_indices()

        for counter, index in enumerate(self.indices):
            self.states[counter] = self._generate_state(index - 1)
            self.future_states[counter] = self._generate_state(index)
        
        return np.transpose(self.states, axes=(0,2,3,1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.future_states, axes=(0,2,3,1)), self.dones[self.indices]
    
# We now start coding the two types of layers comprising the DSN:
# the convolutional and the hidden layer classes. These initialize
# the parameters based on the input and output feature map sizes, 
# and also define the activation functions. For the purpose of this 
# investigation, to keep in line with the original paper, we will 
# use the Relu activation function. 

class ConvolutionalLayer:
    def __init__ (self, Mi, Mo, filter_size = 5, stride = 2, activation_function = tf.nn.relu):
        self.W = tf.Variable(tf.random_normal(shape = (filter_size, filter_size, Mi, Mo)) * np.sqrt(2.0 / (Mi+Mo)))
        self.parameters = [self.W]
        self.b = tf.Variable(np.zeros(Mo, dtype = np.float32))
        self.parameters.append(self.b)
        self.activation_function = activation_function
        self.stride = stride

    def forward(self,X):
        convolution_output = tf.nn.conv2d(X, self.W, strides= [1, self.stride, self.stride, 1], padding = 'SAME')
        convolution_output = tf.nn.bias_add(convolution_output, self.b)
        return self.activation_function(convolution_output)

class HiddenLayer:
    def __init__(self, M1, M2, activation_function = tf.nn.relu, use_bias = True):
        self.W = tf.Variable(tf.random_normal(shape = (M1, M2)) * np.sqrt(2 / (M1+M2)))
        self.parameters = [self.W]
        self.use_bias = use_bias

        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.parameters.append(self.b)

        self.activation_function = activation_function

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.activation_function(a)

# Now, we define the DSN class:

class DSN:
    def __init__ (self, outputs, convolutional_layer_sizes, hidden_layer_sizes, scope):
        
        self.outputs = outputs
        self.scope = scope

        # The scope is necessary in order to differentiate between the main and target networks

        with tf.variable_scope(scope):

            # We first go throught the convolutional layers

            self.convolutional_layers = []
            Mi = 4 # input feature size
            height = frame_size
            width = frame_size

            for Mo, filter_size, stride in convolutional_layer_sizes:
                convolutional_layer = ConvolutionalLayer(Mi, Mo, filter_size, stride)
                self.convolutional_layers.append(convolutional_layer)

                Mi = Mo

                # Following this, we need to calculate the final output size
                # as this will be required for the fully connected layer

                height = int(np.ceil(height / stride)) #simplified version of the original equation
                width = int(np.ceil(width / stride))

            flattened_output_size = height * width * Mi

            # Now, we build the fully connected layer

            self.layers = []

            M1 = flattened_output_size
            for M2 in hidden_layer_sizes:
                layer = HiddenLayer(M1, M2)
                self.layers.append(layer)
                M1 = M2

            final_layer = HiddenLayer(M1, outputs, lambda x:x)
            self.layers.append(final_layer)

            # After this has been done, we collect all the parameters in
            # the newtork as this will be necessary when creating the 
            # seconday network. This will also help in updating the
            # parameters.

            self.parameters = []
            for layer in (self.convolutional_layers + self.layers):
                self.parameters += layer.parameters

            # Having done this, we declare the input and the target
            # tensorflow variables.

            self.X = tf.placeholder(tf.float32, shape = (None, frame_size, frame_size, 4), name = 'X')
            self.G = tf.placeholder(tf.float32, shape = (None,), name = 'G')
            self.actions = tf.placeholder(tf.int32, shape = (None,), name = 'actions')

            # We now calculate the outputs and the cost

            Z = self.X / 255.0

            for layer in self.convolutional_layers:
                Z = layer.forward(Z)

            Z = tf.reshape(Z, [-1, flattened_output_size])

            for layer in self.layers:
                Z = layer.forward(Z)

            Y_hat = Z

            # We set the prediction operation to be Y-hat, as this is
            # the desired model output. 

            self.prediction_operation = Y_hat

            selected_action_values = tf.reduce_sum(
                self.prediction_operation * tf.one_hot(self.actions, outputs),
                reduction_indices= [1]
            )

            cost = tf.reduce_mean(tf.square(self.G - selected_action_values))

            self.cost = cost

            self.training_operation = tf.train.RMSPropOptimizer(2.5e-4, decay= 0.99, epsilon = 1e-3).minimize(cost)

        # This marks the end of the constructor. Now we build a few
        # additonal functions

    # First, we create the set_session function. This is a basic setter.

    def set_session(self, session):
        self.session = session

    def copy_parameters(self, other):

        # This function copies all the parameters from the input
        # network. All the values in the input netork are copied to
        # the corresponding parameters in the other network. We first 
        # get all the parameters in the scope of each network, and then
        # we sort them so that they appear in the same order for both
        # networks. 

        # These are all operations, so we need to run them inside a
        # session

        original_parameters = [variable for variable in tf.trainable_variables() if variable.name.startswith(self.scope)]
        original_parameters = sorted(original_parameters, key= lambda variable: variable.name)

        copied_parameters = [variable for variable in tf.trainable_variables() if variable.name.startswith(other.scope)]
        copied_parameters = sorted(copied_parameters, key= lambda variable: variable.name)

        operations = []
        for param1, param2 in zip (original_parameters, copied_parameters):
            operation = param1.assign(param2)
            operations.append(operation)

        self.session.run(operations)

    # We now define a save and a load function, allowing us to save all
    # the arrays in a single function, and also load the them latter on.

    def save(self):
        parameters = [variable for variable in tf.trainable_variables() if variable.name.startswith(self.scope)]
        parameters = self.session.run(parameters)
        np.savez('BDE_breakout_tensorflow_weights.npz', *parameters)

    def load(self):
        parameters = [variable for variable in tf.trainable_variables() if variable.name.startswith(self.scope)]
        npz = np.load('BDE_breakout_tensorflow_weights.npz')
        operations = []
        for parameter, (_, value) in zip(parameters, npz.iteritems()):
            operations.append(parameter.assign(value))
        self.session.run(operations)

    def predict(self, states):
        return self.session.run(self.prediction_operation, feed_dict = {self.X: states})

    def update(self, states, actions, targets):
        
        # This function runs one iteration of the training operation
        # and also calculates and returns the loss

        loss, _ = self.session.run(
            [self.cost, self.training_operation], 
            feed_dict={
                self.X: states,
                self.G: targets,
                self.actions: actions
            }
        )

        return loss

    def sample_action(self, x, epsilon):

        # This function will be requred for sampling the actions.

        if np.random.random() < epsilon:
            return np.random.choice(self.outputs)
        else:
            return np.argmax(self.predict([x])[0])

# Having finished the constructor class, we then define the a Learn 
# function. This function sets up the update. It takes a batch of data
# from the experience replay memory, calculates the targets with the 
# target network and then passes them back into the main network. 

def learn(network, target_network, experience_replay, gamma, batch_size, epsilon):

    states, actions, rewards, next_states, dones = experience_replay.get_minibatch()

    # The next states are then used to predict the Q values.

    Q_values = target_network.predict(next_states)

    Q_predicted = []
    probabilities = np.random.random(Q_values.shape[0])
    for index in range(Q_values.shape[0]):
        if probabilities[index] < epsilon:
            Q_predicted.append(np.random.choice(Q_values[index]))
        else:
            Q_predicted.append(np.max(Q_values[index]))

    Q_predicted = np.atleast_1d(Q_predicted)

    # Then, using these, we can calculate the targets, or returns (G)

    targets = rewards + np.invert(dones).astype(np.float32) * gamma * Q_predicted

    # Finally, using this data, we call the training optimizer and
    # update the network

    loss = network.update(states, actions, targets)

    return loss

def play_one_episode(environment, session, total_steps, experience_replay, network, target_network, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_minimum):

    # This function goes through one episode of the game.

    start_time = datetime.now()
    total_training_time = 0

    observation = environment.reset()
    transformed_observation = image_transformer.transform_image(observation, session)
    state = np.stack([transformed_observation] * 4, axis= 2) # We stack the same observation 4 times in order to construct our first state, as we have no state to start from

    steps = 0
    total_reward = 0

    done = False

    action_1 = network.sample_action(state, epsilon)
    action_t = action_1

    while not done:
        if total_steps % copy_period == 0:
            target_network.copy_parameters(network)
            print("Copied network parameters to target network. total steps = %s, copy period = %s" % (total_steps, copy_period))
            print("Copied network parameters to target network. total steps = %s, copy period = %s" % (total_steps, copy_period),file=open("outputBDE.txt", "a"))

        observation, reward, done, _ = environment.step(action_t)
        transformed_observation = image_transformer.transform_image(observation, session)
        next_state = update_state(state, transformed_observation)

        total_reward += reward

        # After taking an action, we store the data in our experience
        # replay database, and then we update the network.

        experience_replay.add_experience(transformed_observation, action_t, reward, done)

        # Now, we train the network
        # The time is required in order to give an indication of how long we spend in each training step.

        train_time_start = datetime.now()
        loss = learn(network, target_network, experience_replay, gamma, batch_size, epsilon)

        delta_train_time = datetime.now() - train_time_start

        total_training_time += delta_train_time.total_seconds()

        steps += 1

        state = next_state

        total_steps +=1

        action_t = network.sample_action(state, epsilon)

        epsilon = max(epsilon - epsilon_change, epsilon_minimum)

    return total_steps, total_reward, (datetime.now() - start_time), steps, total_training_time / steps, epsilon

def play_evaluation_episode(environment, session, total_steps, experience_replay, network, target_network, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_minimum):

    # This function goes through one episode of the game.

    observation = environment.reset()
    transformed_observation = image_transformer.transform_image(observation, session)
    state = np.stack([transformed_observation] * 4, axis= 2) # We stack the same observation 4 times in order to construct our first state, as we have no state to start from

    total_evaluation_reward = 0

    action_1 = network.sample_action(state, epsilon)
    action_t = action_1

    done = False

    while not done:

        observation, reward, done, _ = environment.step(action_t)
        transformed_observation = image_transformer.transform_image(observation, session)
        next_state = update_state(state, transformed_observation)

        state = next_state

        action_t = network.sample_action(state, epsilon)

        total_evaluation_reward += reward

    return total_evaluation_reward

def average_returns(x):

    # Returns the average return over the past 100 steps.
    # Gives a more stable looking plot of returns over time. 

    returns = np.zeros(len(x))

    for index in range(len(x)):
        start_point = max(0, index-99)
        returns[index] = float(x[start_point:(index+1)].sum() / (index - start_point+1))
    
    return returns

def main():

    # This represents the main part of the function

    file_name = 'BDE'
    console_file = file_name+ '.txt'

    # We first start by defining the hyperparameters and
    # initializing the variables
    convolutional_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_size = 32
    number_of_episodes = 1000
    total_steps = 0
    experience_replay = Replay_Memory()
    rewards = np.zeros(number_of_episodes)
    mean_evaluation_rewards = []
    standard_deviation_rewards = []

    # The epsilon decays linearly until 0.1:

    epsilon = 1.0
    epsilon_minimum = 0.1
    epsilon_change = (epsilon - epsilon_minimum) / 500000

    # We then create the environment:
    environment = gym.envs.make("Seaquest-v0")
    outputs = environment.action_space.n

    # Next, we create our networks:

    network = DSN(outputs, convolutional_layer_sizes, hidden_layer_sizes, scope="network")

    target_network = DSN(outputs, convolutional_layer_sizes, hidden_layer_sizes, scope="target_network")

    image_transformer = ImageTransformer()

    evaluation_period = 4000
    evaluation_episodes = 100
    final_evaluation_episodes = 1000

    with tf.Session() as session:
        network.set_session(session)
        target_network.set_session(session)
        session.run(tf.global_variables_initializer())

        # First, we populate the experience replay by taking completely
        # random actions

        print("Creating experience replay buffer...")

        observation = environment.reset()

        for i in range(min_train_experiences):
            action = np.random.choice(outputs)
            observation, reward, done, _ = environment.step(action)
            transformed_observation = image_transformer.transform_image(observation, session)
            experience_replay.add_experience(transformed_observation, action, reward, done)

            if done:
                observation = environment.reset()

        # Now we start playing

        start_time = datetime.now()

        for episode in range (number_of_episodes):

            total_steps, total_reward, duration, steps, time_per_step, epsilon = play_one_episode (environment, session, total_steps, experience_replay, network, target_network, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_minimum)

            rewards[episode] = total_reward

            average_last_100_rewards = rewards[max(0, episode-100):(episode+1)].mean()

            print(
                "Episode:", episode,
                "Duration:", duration,
                "Number of steps:", steps,
                "Reward:", total_reward,
                "Training time per step:", "%.3f" % time_per_step,
                "Average Reward (last 100 episodes):" "%.3f" % average_last_100_rewards,
                "Epsilon:", "%.3f" % epsilon
            )

            print(
                "Episode:", episode,
                "Duration:", duration,
                "Number of steps:", steps,
                "Reward:", total_reward,
                "Training time per step:", "%.3f" % time_per_step,
                "Average Reward (last 100 episodes):" "%.3f" % average_last_100_rewards,
                "Epsilon:", "%.3f" % epsilon,
                file=open("outputBDE.txt","a")
            )

            if episode % evaluation_period == 0 and episode != 0:

                evaluation_rewards = np.empty(evaluation_episodes)

                for evaluation_episode in range(evaluation_episodes):

                    evaluation_reward = play_evaluation_episode (environment, session, total_steps, experience_replay, network, target_network, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_minimum)
                    evaluation_rewards[evaluation_episode] = evaluation_reward

                mean_evaluation_reward = np.mean(evaluation_rewards)
                standard_deviation_reward = np.std(evaluation_rewards)

                mean_evaluation_rewards.append(mean_evaluation_reward)
                standard_deviation_rewards.append(standard_deviation_reward)

            if episode == number_of_episodes-1:

                evaluation_rewards = np.empty(evaluation_episodes)

                for evaluation_episode in range(evaluation_episodes):

                    evaluation_reward = play_evaluation_episode (environment, session, total_steps, experience_replay, network, target_network, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_minimum)
                    evaluation_rewards[evaluation_episode] = evaluation_reward

                mean_evaluation_reward = np.mean(evaluation_rewards)
                standard_deviation_reward = np.std(evaluation_rewards)

                mean_evaluation_rewards.append(mean_evaluation_reward)
                standard_deviation_rewards.append(standard_deviation_reward)

            sys.stdout.flush()

        final_rewards = np.empty(final_evaluation_episodes)

        for evaluation_episode in range(final_evaluation_episodes):

            evaluation_reward = play_evaluation_episode (environment, session, total_steps, experience_replay, network, target_network, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_minimum)
            final_rewards[evaluation_episode] = evaluation_reward

        mean_final_reward = np.mean(final_rewards)
        standard_deviation_final_reward = np.std(final_rewards)
        
        print("Total duration:", datetime.now() - start_time)
        print("Total duration:", datetime.now() - start_time, file=open("outputBDE.txt", "a"))

        network.save()

        # Finally, we plot the smoothed returns:

        average_rewards = average_returns(rewards)
        plt.plot(rewards, label= 'Original')
        plt.plot(average_rewards, label='Averaged (last 100)')
        plt.legend()
        plt.show()
        plt.savefig("Rewards.png")

    with open('rewards'+file_name+'.csv', "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(rewards)

    with open('eval'+file_name+'.csv', "a") as output_file:
        writer = csv.writer(output_file)        
        writer.writerow(mean_evaluation_rewards)

    with open('std_eval'+file_name+'.csv', "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(standard_deviation_rewards)

    with open('final'+file_name+'.csv', "a") as output_file:
        writer = csv.writer(output_file)   
        writer.writerow([mean_final_reward])

    with open('std_final'+file_name+'.csv', "a") as output_file:
        writer = csv.writer(output_file)   
        writer.writerow([standard_deviation_final_reward])

    with open('run_avg'+file_name+'.csv', "a") as output_file:
        writer = csv.writer(output_file)   
        writer.writerow(average_rewards)

    with open('final_rewards'+file_name+'.csv', "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(final_rewards)

if __name__ == '__main__':
    main()