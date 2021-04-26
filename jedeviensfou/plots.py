import matplotlib.pyplot as plt
import numpy as np 

rewards = np.load("saves/rewards.npy")
plt.plot(rewards)
plt.savefig("graphs/rewards_ddqn")

plt.clf()
epsilon = np.load("saves/epsilon.npy")
plt.plot(epsilon)
plt.savefig("graphs/epsilon_ddqn")