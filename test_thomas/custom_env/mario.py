import gym

import custom_env

env_name = "CustomEnv-v0"

env = gym.make(env_name)

env.reset()

state, reward, done, info = env.step(env.action_space.sample())

i = 0

while True:
    i += 1
    if not i % 1000:
        print(i)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
