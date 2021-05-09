from deep_q_learning_tf1_fast import Agent
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gym
import sys
import gym_super_mario_bros
import wrappers_gym
import numpy as np
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from array2gif import write_gif
import time
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

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

from gym.wrappers.monitoring.video_recorder import VideoRecorder

if __name__=="__main__":
    #tf.compat.v1.disable_eager_execution()
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = wrappers_gym.wrapper(env, clip_reward=True)
    env_ref = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env_ref = JoypadSpace(env_ref, RIGHT_ONLY)

    if len(sys.argv) > 2:
        video_recorder = VideoRecorder(env_ref, sys.argv[2], enabled=True)
    else:
        video_recorder = VideoRecorder(env_ref, './result.mp4', enabled=False)
    agent = Mario(gamma=0.99, 
                epsilon=1.0, 
                learning_rate=0.01,
                input_dims=env.observation_space.shape,
                nb_actions=env.action_space.n,
                learning_frequency=1,
                memory_size=10000000,
                batch_size=64,
                burnin=1,
                epsilon_end=0.01,
                epsilon_decay=0.99,
                demo_mode=True,
                model_path=sys.argv[1])

    done = False
    score = 0
    state = env.reset()
    env_ref.reset()
    skipped_frames = 4

    pause_at_end = False
    
    while not done:
        action = agent.choose_action(state)
        frame = env.render()
        next_state, reward, done, info = env.step(action=action)
        if not done:
            for i in range(skipped_frames):
                env_ref.step(action=action)
                video_recorder.capture_frame()
        else:
            env_ref.step(action=action)
            video_recorder.capture_frame()
        score += reward
        state = next_state

    env_ref.close()
    video_recorder.close()
    print(f"score : {score:.2f}")

    env.close()

    