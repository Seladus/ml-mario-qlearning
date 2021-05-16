import time

import numpy as np
from nsmm3dq.agent import Agent


class QLearning:
    def __init__(
        self,
        env,
        agent_class,
        n_episodes=25000,
        max_step_in_episode=5000,
        savepath="saves",
        history_update_freq=20,
        agent_params={},
    ):

        input_dims = env.observation_space.shape
        nb_actions = env.action_space.n

        self._env = env
        self._agent = agent_class(input_dims, nb_actions, **agent_params)
        self._n_episodes = n_episodes
        self._max_step_in_episode = max_step_in_episode
        self._savepath = savepath
        self._history_update_freq = history_update_freq

    def start(self):
        rewards = []
        epsilons = []
        for episode in range(self._n_episodes + 1):
            done = False
            score = 0
            count = 0
            state = self._env.reset()
            start_time = time.time()
            while not done and count < self._max_step_in_episode:
                action = self._agent.choose_action(state)
                next_state, reward, done, info = self._env.step(action)
                score += reward
                self._agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                self._agent.learn()
                count += 1

            self._agent.save_model(episode, path=self._savepath)

            epsilons.append(self._agent.epsilon)
            rewards.append(score)

            average_score = np.mean(rewards[-100:])

            if episode % self._history_update_freq == 0:
                np.save(f"{self._savepath}/rewards", np.array(rewards))
                np.save(f"{self._savepath}/epsilon", np.array(epsilons))

            if episode % self._history_update_freq == 0:
                print(
                    f"episode : {episode} - epsilon : {self._agent.epsilon:.2f} - frame count : {self._agent.step} - frame/sec {np.round(count / (time.time() - start_time))} - score {score:.2f} - average score : {average_score:.2f}"
                )

        self._agent.save_model(episode, path=self._savepath)
        np.save(f"{self._savepath}/rewards", np.array(rewards))
        np.save(f"{self._savepath}/epsilon", np.array(epsilons))
