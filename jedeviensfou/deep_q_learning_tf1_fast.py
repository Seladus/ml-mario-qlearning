import numpy as np
import random
#import tensorflow as tf 
import tensorflow.compat.v1 as tf 
#from tensorflow import keras
from tensorflow.compat.v1 import keras
#from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.optimizers import Adam
from collections import deque
import gym

tf.disable_v2_behavior()

def copy_weights(source, destination):
    destination.set_weights(source.get_weights())

class ExperienceBuffer:
    def __init__(self, buffer_size):
        self.size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def add_experience(self, experience):
        self.buffer.append(experience)
    
    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)

class Agent:
    def __init__(self, 
                learning_rate,
                gamma,
                nb_actions,
                epsilon,
                batch_size,
                input_dims,
                burnin,
                learning_frequency=4,
                epsilon_decay=0.99,
                epsilon_end=0.01,
                memory_size=1000000,
                save_frequency=100,
                model_name='dqn_network',
                demo_mode=False,
                model_path="",
                double_q_learning=False,
                target_update_frequency=1000,
                is_epsilon_decaying=False,
                is_epsilon_decaying_linear=False):
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
                self.update_target()


    def decrease_epsilon(self):
        if self.is_epsilon_decaying_linear:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_end, self.epsilon)

    def update_target(self):
        if self.double_q:
            print("[INFO] updating target network...")
            copy_weights(self.q_eval, self.q_target)

    def create_model(self, learning_rate, input_dims, nb_actions):
        model = keras.Sequential([
            keras.layers.Input(self.input_dims),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(nb_actions, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='huber_loss')
        return model
    
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
            self.decrease_epsilon()
        
        self.step += 1
        return action
    
    def learn(self):
        if len(self.memory.buffer) < self.burnin:
            return
        if self.step % self.update_frequency == 0:
            self.update_target()
        if self.learning_step % self.learning_frequency != 0:
            self.learning_step += 1
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        if self.double_q:
            # Predict Q(s,a) and Q(s',a') given the batch of states
            #q_values_state = self.q_eval(states).numpy()
            q_values_state = self.q_eval.predict_on_batch(states)
            #q_values_next_state = self.q_eval(next_states).numpy()
            q_values_next_state = self.q_eval.predict_on_batch(next_states)

            #Initialize target
            targets = q_values_state
            updates = np.zeros(rewards.shape)

            action = np.argmax(q_values_next_state, axis=1) # argmax(Q(S_t+1, a))
            q_values_next_state_target = self.q_target.predict_on_batch(next_states)
            #updates = rewards + (1 - dones) * self.gamma * q_values_next_state_target[range(self.batch_size), action]
            updates = rewards + (1 - dones) * self.gamma * q_values_next_state_target[np.arange(0, self.batch_size), action]
            targets[range(self.batch_size), actions] = updates
            self.q_eval.train_on_batch(states, targets)



        else:
            q_next = self.q_eval.predict(next_states).max(axis=1)
            targets = self.q_eval.predict(states)
            targets[range(self.batch_size), actions] = rewards + (1 - dones) * self.gamma * q_next 
            self.q_eval.train_on_batch(states, targets)
        self.learning_step = 0

if __name__=="__main__":
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    learning_rate = 0.001
    episodes = 800
    agent = Agent(gamma=0.99, 
                epsilon=1.0, 
                learning_rate=learning_rate,
                input_dims=env.observation_space.shape,
                nb_actions=env.action_space.n,
                learning_frequency=1,
                memory_size=10000000,
                batch_size=64,
                burnin=64,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                double_q_learning=True,
                target_update_frequency=1000)
    rewards = []
    epsilon_history = []

    for episode in range(episodes):
        done = False
        score = 0
        state = env.reset()
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn()
        agent.epsilon = max(agent.epsilon*agent.epsilon_decay, agent.epsilon_end)
        
        agent.save_model(episode)

        epsilon_history.append(agent.epsilon)
        rewards.append(score)
        average_score = np.mean(rewards[-100:])
        print(f"episode : {episode} - epsilon : {agent.epsilon:.2f} - score {score:.2f} - average score : {average_score:.2f}")
    
    agent.save_model("final")
    np.save("saves/rewards", np.array(rewards))
    np.save("saves/epsilon", np.array(epsilon_history))


            
