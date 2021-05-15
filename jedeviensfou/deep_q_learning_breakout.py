import gym
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from deep_q_learning_tf1_fast import Agent

class Mario(Agent):
    def create_model(self, learning_rate, input_dims, nb_actions):
        model = keras.models.Sequential([
            keras.layers.Input(input_dims),
            keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(nb_actions, activation='linear')
        ])

        model.compile(optimizer=Adam(learning_rate), loss='huber_loss')
        return model

import numpy as np
import gym_super_mario_bros
import wrappers_gym
from signal import signal, SIGINT
from gym.wrappers import FrameStack
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import tensorflow as tf

import time

import matplotlib.pyplot as plt
    
if __name__=="__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    env = gym_super_mario_bros.make('Breakout-v0')
    #env = SkipFrame(env, 4)
    #env = GrayScaleObservation(env)
    #env = ResizeObservation(env, (84, 84))
    #env = FrameStack(env, 4)
    env = wrappers_gym.wrapper(env)

    episodes = 10000
    max_steps_in_episode = 10000

    save_rewards_freq = 30
    path_saves = "saves/models/ddqn_breakout"
    path_rewards = "saves/models/ddqn_breakout"

    agent = Mario(
        gamma=0.99,
        epsilon=1.0, 
        learning_rate=0.00025,
        input_dims=(84, 84, 4),
        nb_actions=env.action_space.n,
        learning_frequency=4,
        burnin=32,
        memory_size=50000,
        batch_size=32,
        epsilon_end=0.1,
        epsilon_decay=0.9/300000,
        double_q_learning=True,
        target_update_frequency=2500,
        is_epsilon_decaying=True,
        is_epsilon_decaying_linear=True,
        save_frequency=50,
        model_name="breakout-ddqn")

    rewards = []
    epsilon_history = []
    start = time.time()
    for episode in range(episodes):
        done = False
        score = 0
        count = 0
        state = env.reset()
        
        while not done and count < max_steps_in_episode:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn()
            count += 1
        
        agent.save_model(episode, path=path_saves)
        if episode % save_rewards_freq == 0:
            np.save(f"{path_rewards}/rewards", np.array(rewards))
            np.save(f"{path_rewards}/epsilon", np.array(epsilon_history))
        epsilon_history.append(agent.epsilon)
        rewards.append(score)
        average_score = np.mean(rewards[-100:])
        print(f"episode : {episode} - epsilon : {agent.epsilon:.2f} - frame count : {agent.step} - score {score:.2f} - average score : {average_score:.2f}")
    
    end = time.time()
    print(f"Time elapsed : {end - start}")
    agent.save_model(episode, path=path_saves)
    np.save(f"{path_rewards}/rewards", np.array(rewards))
    np.save(f"{path_rewards}/epsilon", np.array(epsilon_history))

