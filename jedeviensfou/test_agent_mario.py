from deep_q_learning_tf2_fast import Agent
import tensorflow as tf
import gym
import sys
import gym_super_mario_bros
import wrappers_gym
import numpy as np
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from array2gif import write_gif

if __name__=="__main__":
    #tf.compat.v1.disable_eager_execution()
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = wrappers_gym.wrapper(env, clip_reward=True)
    model = tf.keras.models.load_model('saves/models/ddqn_mario_tf2_fast/marioQ_0.h5')
    agent = Agent(gamma=0.99, 
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
    frames = []
    frame_counter = 0
    frame_save = 10

    savefig = False
    pause_at_end = True
    
    agent.q_eval.summary()

    while not done:
        action = agent.choose_action(state)
        #frame = env.render(mode='rgb_array')
        frame = env.render()
        if frame_counter % frame_save == 0:
            frames.append(frame)
        next_state, reward, done, info = env.step(action=action)
        score += reward
        state = next_state

        frame_counter += 1
        
    print(f"score : {score:.2f}")
    if pause_at_end:
        input()

    env.close()

    if savefig:
        write_gif(np.array(frames), "lunar.gif", fps=10)

    