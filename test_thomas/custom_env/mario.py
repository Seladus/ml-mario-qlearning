import gym

import custom_env

env_name = "CustomEnv-v0"

env = gym.make(env_name)

env.reset()

state, reward, done, info = env.step(env.action_space.sample())

print(state, reward, done, info)
