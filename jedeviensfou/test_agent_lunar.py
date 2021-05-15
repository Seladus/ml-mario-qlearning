from deep_q_learning import Agent
import tensorflow as tf
import gym
import sys
import numpy as np
from array2gif import write_gif

if __name__=="__main__":
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, 
                epsilon=1.0, 
                learning_rate=0.01,
                input_dims=env.observation_space.shape,
                nb_actions=env.action_space.n,
                learning_frequency=1,
                memory_size=10000000,
                batch_size=64,
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
        
    while not done:
        action = agent.choose_action(state)
        frame = env.render(mode='rgb_array')
        if frame_counter % frame_save == 0:
            frames.append(frame)
        next_state, reward, done, info = env.step(action)
        score += reward
        state = next_state

        frame_counter += 1
        
    print(f"score : {score:.2f}")
    if pause_at_end:
        input()

    env.close()

    if savefig:
        write_gif(np.array(frames), "lunar.gif", fps=10)

    