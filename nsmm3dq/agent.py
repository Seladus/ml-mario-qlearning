import random
from collections import deque

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


class ExperienceBuffer:
    def __init__(self, buffer_size):
        self.size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)


class Agent:
    @staticmethod
    def copy_weights(source, destination):
        destination.set_weights(source.get_weights())

    def __init__(
        self,
        input_dims,
        nb_actions,
        learning_rate=0.00025,
        gamma=0.90,
        epsilon=1.0,
        batch_size=32,
        burnin=100000,
        learning_frequency=1,
        epsilon_decay=0.999999,
        epsilon_end=0.01,
        memory_size=1000000,
        save_frequency=50,
        model_name="dqn_network",
        demo_mode=False,
        model_path="",
        double_q_learning=True,
        target_update_frequency=10000,
        is_epsilon_decaying=True,
        is_epsilon_decaying_linear=False,
    ):
        self.actions = [i for i in range(nb_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.memory = ExperienceBuffer(memory_size)
        self.learning_frequency = learning_frequency
        self.learning_step = 0
        self.save_frequency = save_frequency
        self.name = model_name
        self.double_q = double_q_learning
        self.step = 0
        self.update_frequency = target_update_frequency
        self.is_epsilon_decaying = is_epsilon_decaying
        self.is_epsilon_decaying_linear = is_epsilon_decaying_linear
        self.burnin = burnin

        if demo_mode:
            self.q_eval = self.create_model(learning_rate, input_dims, nb_actions)
            self.q_eval.load_weights(model_path)
            self.epsilon = 0.0
        else:
            self.q_eval = self.create_model(learning_rate, input_dims, nb_actions)
            if self.double_q:
                self.q_target = self.create_model(learning_rate, input_dims, nb_actions)
                self._update_target()

    def _decrease_epsilon(self):
        if self.is_epsilon_decaying_linear:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_end, self.epsilon)

    def _update_target(self):
        if self.double_q:
            Agent.copy_weights(self.q_eval, self.q_target)

    def create_model(self, learning_rate, input_dims, nb_actions):
        raise NotImplementedError()

    def save_model(self, episode_count, path="saves/models"):
        if episode_count % self.save_frequency == 0:
            self.q_eval.save(f"{path}/{self.name}_{episode_count}.h5")

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.add_experience(experience=(state, action, reward, new_state, done))

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state = np.array([observation])
            q_values = self.q_eval.predict_on_batch(state)
            action = np.argmax(q_values)

        if self.is_epsilon_decaying and self.step > self.burnin:
            self._decrease_epsilon()

        self.step += 1
        return action

    def learn(self):
        if len(self.memory.buffer) < self.burnin:
            return
        if self.step % self.update_frequency == 0:
            self._update_target()
        if self.learning_step % self.learning_frequency != 0:
            self.learning_step += 1
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        if self.double_q:
            # Predict Q(s,a) and Q(s',a') given the batch of states
            q_values_state = self.q_eval.predict_on_batch(states)
            q_values_next_state = self.q_eval.predict_on_batch(next_states)

            # Initialize target
            targets = q_values_state
            updates = np.zeros(rewards.shape)

            action = np.argmax(q_values_next_state, axis=1)  # argmax(Q(S_t+1, a))
            q_values_next_state_target = self.q_target.predict_on_batch(next_states)
            updates = (
                rewards
                + (1 - dones)
                * self.gamma
                * q_values_next_state_target[np.arange(0, self.batch_size), action]
            )
            targets[range(self.batch_size), actions] = updates
            self.q_eval.train_on_batch(states, targets)

        else:
            q_next = self.q_eval.predict(next_states).max(axis=1)
            targets = self.q_eval.predict(states)
            targets[range(self.batch_size), actions] = (
                rewards + (1 - dones) * self.gamma * q_next
            )
            self.q_eval.train_on_batch(states, targets)
        self.learning_step = 0
