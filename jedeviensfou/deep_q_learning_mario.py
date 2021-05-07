import gym
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from deep_q_learning import Agent

class Mario(Agent):
    def create_model(self, learning_rate, input_dims, nb_actions):
        model = keras.models.Sequential([
            keras.layers.Input(input_dims),
            keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
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

import time

import matplotlib.pyplot as plt
    
if __name__=="__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    #env = SkipFrame(env, 4)
    #env = GrayScaleObservation(env)
    #env = ResizeObservation(env, (84, 84))
    #env = FrameStack(env, 4)
    env = wrappers_gym.wrapper(env)

    episodes = 10000

    agent = Mario(
        gamma=0.90,
        epsilon=1.0, 
        learning_rate=0.00025,
        input_dims=(84, 84, 4),
        nb_actions=env.action_space.n,
        learning_frequency=3,
        burnin=50000,
        memory_size=50000,
        batch_size=32,
        epsilon_end=0.1,
        epsilon_decay=0.9999975,
        double_q_learning=True,
        target_update_frequency=10000,
        is_epsilon_decaying=True,
        save_frequency=100,
        model_name="marioQ")

    rewards = []
    epsilon_history = []
    start = time.time()
    for episode in range(episodes):
        done = False
        score = 0
        count = 0
        state = env.reset()
        #state = np.transpose(state)
        
        while not done and count < 5000:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            #next_state = np.transpose(next_state)
            score += reward
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn()
            count += 1
        #agent.epsilon = max(agent.epsilon*agent.epsilon_decay, agent.epsilon_end)
        
        agent.save_model(episode, path="saves/models/ddqn_cuda_local")

        epsilon_history.append(agent.epsilon)
        rewards.append(score)
        average_score = np.mean(rewards[-100:])
        print(f"episode : {episode} - epsilon : {agent.epsilon:.2f} - frame count : {agent.step} - score {score:.2f} - average score : {average_score:.2f}")
    
    end = time.time()
    print(f"Time elapsed : {end - start}")
    #agent.save_model(episodes)
    np.save("saves/models/ddqn_cuda_local/rewards", np.array(rewards))
    np.save("saves/models/ddqn_cuda_local/epsilon", np.array(epsilon_history))


    

