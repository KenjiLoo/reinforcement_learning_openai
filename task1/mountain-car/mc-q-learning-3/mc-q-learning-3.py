#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install pyglet==1.5.0')


# In[2]:


import sys
import gym
import pylab
import random
import numpy as np
import matplotlib.pyplot as plt
# from IPython import display
from collections import deque
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import model_from_json
from Agent import MountainCarAgent
from keras.layers import LeakyReLU
from keras.initializers import he_normal
# from keras.initializers import RandomNormal

from keras.optimizers import SGD

# In[ ]:


# In[3]:


# to store rewards in each episode
rewards_per_episode, episodes = [], []

# In[4]:


env = gym.make('MountainCar-v0', render_mode='human')
env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    env.step(action)
    env.render()
env.close()

env.reset()

# In[5]:


state_size = env.observation_space.shape[0]  # equal to 4 in case of cartpole
action_size = env.action_space.n

# In[6]:


agent = MountainCarAgent(action_size, state_size)

# In[7]:


n_episodes = 40
update_counter = 0
epsilon = 1

# In[8]:


#### simulation starts ####
for episode in range(n_episodes):

    done = False
    score = 0
    # reset at the start of each episode
    state, _ = env.reset()

    while not done:
        env.render()

        agent.epsilon = 0.01 + (1 - 0.01) * np.exp(-0.001 * update_counter)
        update_counter = update_counter + 1
        # get action for the current state and take a step in the environment
        action = agent.get_action(state)
        next_state, reward, done, info, _ = env.step(action)

        reward = 100 * ((np.sin(3 * next_state[0]) * 0.0025 + 0.5 * next_state[1] * next_state[1]) - (
                    np.sin(3 * state[0]) * 0.0025 + 0.5 * state[1] * state[1]))

        if next_state[0] >= 0.5:
            reward += 1

        # save the sample <s, a, r, s', done> to the replay memory
        agent.append_sample(state, action, reward, next_state, done)

        # train after each step
        agent.train_model()

        # add reward to the total score of this episode
        score += reward
        state = next_state
        if done:
            agent.update_target_model()

    # store total reward obtained in this episode
    rewards_per_episode.append(score)
    episodes.append(episode)

    # terminate if no major learning for previous 5 episodes
    if np.mean(rewards_per_episode[-7:]) > 1.5:
        break;

    # every episode:
    print("episode {0}, reward {1}".format(episode, score))
    # every few episodes:
    if episode > 1 and episode % 5 == 0:
        # save model weights
        agent.save_model_weights(name="model_weights.h5")

#### simulation complete ####


# In[9]:


# save stuff as pickle
# def save_pickle(obj, name):
#     with open(name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# In[10]:


# save_pickle(rewards_per_episode, "rewards_per_episode")


# In[11]:


# plot results
# with open('rewards_per_episode.pkl', 'rb') as f:
#     rewards_per_episode = pickle.load(f)


# In[12]:


plt.plot(list(range(len(rewards_per_episode))), rewards_per_episode)
plt.xlabel("episode number")
plt.ylabel("reward per episode")

# In[13]:


print("Average reward of last 100 episodes is {0}".format(np.mean(rewards_per_episode[-100:])))


# In[14]:


def play_game(ml_model, games=10):
    """
    Play te Game
    :param ml_model:
    :param games:
    :return:
    """

    for i_episode in range(games):

        # Define Reward Var
        episode_reward = 0

        # Reset Env for the Game
        observation, _ = env.reset()

        while True:
            render = env.render()

            # Predict Next Movement
            current_action_pred = ml_model.predict(observation.reshape(1, 2))[0]

            # Define Movement
            current_action = np.argmax(current_action_pred)

            # Make Movement
            observation, reward, done, info, _ = env.step(current_action)

            # Update Reward Value
            #             episode_reward += compute_reward(observation[[0]])

            if done:
                print(f"Episode finished after {i_episode + 1} steps", end='')
                break


# In[15]:


def build_model():
    init = he_normal(seed=1000)
    #         init = RandomNormal(mean=0., stddev=1.)
    model = Sequential()
    model.add(Dense(48, input_dim=2, kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(64, kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.1))

    #         model.add(Dense(self.action_size, activation=LeakyReLU(alpha=0.1)))
    model.add(Dense(3, activation=LeakyReLU(alpha=0.1), kernel_initializer=init))
    model.compile(loss="mean_squared_error",
                  optimizer=Adam(lr=0.0001))

    return model



mymodel = build_model()
mymodel.load_weights("./model_weights.h5")


def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=2))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(3, activation=LeakyReLU(alpha=0.1)))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.02))

    return model


time = np.arange(0, 30000)
epsilon = []
for i in range(0, 30000):
    epsilon.append(0.01 + (1 - 0.01) * np.exp(-0.0001 * i))
#     epsilon.append(0.001 + (1 - 0.001) * np.exp(-0.01*i))

plt.plot(time, epsilon)
plt.show()
