from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import gym

import numpy as np
import matplotlib.pyplot as plt

nb_actions = len(gym_super_mario_bros.actions.SIMPLE_MOVEMENT)

env = gym.make("SuperMarioBros-v0")

# Initializing hyper-parameters
max_epsilon = 1 # Initial value for epsilon greedy exploration
min_epsilon = 0.01 # Final value for epsilon greedy exploration
epsilon_decay_steps = 500000
gamma = 0.99 # Coefficient d'actualisation
batch_size = 32
agent_history_length = 4
target_network_update_frequency = 500

learning_rate = 0.00025
gradient_momentum = 0.95
squared_gradient_momentum = 0.95
min_squared_gradient = 0.01

state_dim = 2048
N_start = 1000
N = 10000 # Capacity of replay memory

# Initializing network
import tensorflow as tf
from tensorflow.keras.models import clone_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Input, Flatten, AveragePooling2D, AveragePooling3D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

def build_model(name='Q'):
    Q_star = Sequential(name=name)
    Q_star.add(Input((2048, )))
    Q_star.add(Dense(256, activation='relu'))
    Q_star.add(Dense(nb_actions, activation='linear'))
    return Q_star

def copy_weights(source, destination):
    destination.set_weights(source.get_weights())

def get_initialized_networks():
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=gradient_momentum)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    Q_star = build_model('Q_star')
    Q_star_minus = build_model('Q_star_minus')

    Q_star.compile(optimizer=optimizer,
        loss=loss_object,
        metrics=['accuracy'])
    copy_weights(Q_star, Q_star_minus)
    return Q_star, Q_star_minus

Q_star, Q_star_minus = get_initialized_networks()
Q_star.summary()

from collections import namedtuple
from collections import deque

Transition = namedtuple('Transition', ["state", "action", "reward", "next_state", "done"])

# resetting environnement state
env.reset()
new_state = env.ram

def add_to_replay(D, transition):
        D.append(transition)

D = deque()

# starting to build experience replay from random behavior
for i in range(N_start):
    if not i % 200:
        print(f"{i} basic experiences.")
    action = env.action_space.sample()
    old_state = new_state
    state, reward, done, info = env.step(action)
    new_state = env.ram
    D.append(Transition(old_state, action, reward, new_state, done))

import random

# Training the model now that our experience replay is partially filled
nb_episodes = 10000
episode_rewards = []
best_episode_reward = 0

epsilons = np.linspace(min_epsilon, max_epsilon, epsilon_decay_steps)
frame_counter = 0

for i in range(nb_episodes):
    if not i % 200:
        print(f"{i} new experiences.")
    env.reset()
    new_state = env.ram
    loss = None
    r_sum = 0
    mean_episode_reward = 0

    if episode_rewards:
        mean_episode_reward = np.mean(episode_rewards)
    if best_episode_reward < mean_episode_reward:
        best_episode_reward = mean_episode_reward

    done = False
    while not done:
        # Get epsilon for this step => epsilon greedy policy
        epsilon = epsilons[-frame_counter if frame_counter < epsilon_decay_steps else 0]
        # Update target network Q_star_minus
        if frame_counter % target_network_update_frequency == 0:
            copy_weights(Q_star, Q_star_minus)
        print("\r Epsilon ({}) ReplayMemorySize : ({}) rSum: ({}) best_epi_reward: ({}) OptiStep ({}) @ Episode {}/{}, loss: {}".format(epsilon, len(D), mean_episode_reward, best_episode_reward, frame_counter, i + 1, nb_episodes, loss), end="")

        # Select action using epsilon greedy policy
        random_value = random.random()
        if random_value < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_star.predict(new_state.reshape(1, state_dim)))

        _, reward, done, info = env.step(action)
        old_state = new_state
        new_state = env.ram
        r_sum += reward

        add_to_replay(D, Transition(old_state, action, reward, new_state, done))

        #TODO : learning
