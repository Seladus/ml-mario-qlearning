from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import matplotlib.pyplot as plt

nb_actions = len(gym_super_mario_bros.actions.SIMPLE_MOVEMENT)

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

state_dim = 118
N_start = 5000
N = 10000 # Capacity of replay memory

# Initializing network
import tensorflow as tf
from tensorflow.keras.models import clone_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Input, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

def build_model(name='Q'):
    Q_star = Sequential(name=name)
    Q_star.add(Conv2D(32, (8, 4), activation='relu', input_shape=(240, 256, 3)))
    Q_star.add(Conv2D(64, (4, 2), activation='relu', input_shape=(240, 256, 3)))
    Q_star.add(Conv2D(64, (3, 1), activation='relu', input_shape=(240, 256, 3)))
    Q_star.add(Flatten())
    Q_star.add(Dense(512, activation='relu'))
    Q_star.add(Dense(nb_actions, activation='relu'))
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