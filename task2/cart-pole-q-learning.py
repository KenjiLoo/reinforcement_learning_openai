#--------------------------------------------------------#
import gym
import numpy as np
import random
import math
import time
# from time import sleep
import gym
import numpy as np
import random
import math
from time import sleep
import matplotlib.pyplot as plt
#--------------------------------------------------------#
# Defining the simulation related constants
env = gym.make('CartPole-v0')
np.random.seed(int(time.time())) # Random num generator
NUM_BUCKETS = (1, 1, 6, 3)  # Number of (bucket) per state dimension
NUM_ACTIONS = env.action_space.n  # (left, right)
NUM_TRAIN_EPISODES = 1000  # 1000
NUM_TEST_EPISODES = 1
MAX_TRAIN_T = 250
MAX_TEST_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
VERBOSE = False
ACTION_INDEX = len(NUM_BUCKETS) # Index of the action
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,)) # Creating a Q-Table for each state-action pair
MIN_EXPLORE_RATE = 0.01  # Minimum epsilon
MIN_LEARNING_RATE = 0.01 # Minimum alpha
TEST_RAND_PROB = 0.2

# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = (-0.5, 0.5)
STATE_BOUNDS[3] = (-math.radians(50), math.radians(50))

# Defining variables for e-greedy policy
ALPHA = 0.1  # Step size for gradient descent
w = np.zeros((4, 2))  # Initialize weights
timesteps = [] # Collects timesteps info
#--------------------------------------------------------#

"""
(Function) 'approx' performs linear approximation to expected returns
@param weights -> takes in the weights info
@param observation -> observation space
@param action -> state action value
"""
def approx(weights, observation, action):
    return np.dot(observation, weights)[action]

#--------------------------------------------------------#

"""
(Function) 'policy' is the random or learned policy, selected by epsilon
@param env -> takes in the environment
@param weights -> takes in the weights info
@param observation -> observation space
@param epsilon -> e-greedy value
"""
# Random or Learned Policy, selected by epsilon
def policy(env, weights, observation, epsilon):
    actions = [0, 1]
    if np.random.rand() < epsilon:
        return random.choice(actions)
    qs = []
    for action in actions:
        qs.append(approx(weights, observation, action))
    return np.argmax(qs)

#--------------------------------------------------------#

"""
(Function) 'train' performs training on the agent
"""
def train():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging
    num_train_streaks = 5

    for episode in range(NUM_TRAIN_EPISODES):
        # Update learning streak and state bounds conditions
        if (episode == (NUM_TRAIN_EPISODES / 2)):
            learning_rate = 0.05
            STATE_BOUNDS[3] = (-math.radians(40), math.radians(40))

        obv, _ = env.reset() # Reset the environment
        state_0 = state_to_bucket(obv) # The initial state

        for t in range(MAX_TRAIN_T):
            env.render()
            action = policy(env, w, obv, explore_rate) # Action defined based on policy\
            obv, reward, done, _, _ = env.step(action)
            state = state_to_bucket(obv) # Observe the result

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (
                    reward + discount_factor * (best_q) - q_table[state_0 + (action,)])
            state_0 = state # Setup for the next iteration

            if done:
                print("Episode:%d finished after: %f time steps" % (episode, t))
                timesteps.append(t)
                if (t >= SOLVED_T):
                    num_train_streaks += 1
                else:
                    num_train_streaks = 0
                break

        # It's considered done when it's solved over 120 times consecutively
        if num_train_streaks > STREAK_TO_END:
            print("Solved!")
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

#--------------------------------------------------------#

"""
(Function) 'test' tests the agent on how many timesteps 
it can achieve after training
"""
def test():
    num_test_streaks = 0

    for episode in range(NUM_TEST_EPISODES):
        obv, _ = env.reset() # Reset the environment
        state_0 = state_to_bucket(obv) # The initial state
        tt = 0
        done = False

        while ((abs(obv[0]) < 2.4) & (abs(obv[2]) < 45)):
            tt += 1
            env.render()
            action = policy(env, w, obv, MIN_EXPLORE_RATE) # Selects action
            obv, reward, done, _, _ = env.step(action) # Execute the action
            state_0 = state_to_bucket(obv) # Observe the result
            print("Test episode %d; time step %f." % (episode, tt))

#--------------------------------------------------------#

"""
(Function) 'get_explore_rate' returns epsilon for e-greedy policy
@param t -> number of episode
"""
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t + 1) / 25)))

#--------------------------------------------------------#

"""
(Function) 'get_learning_rate' returns alpha for adaptive learning rate
@param t -> number of episode
"""
def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))

#--------------------------------------------------------#

"""
(Function) 'state_to_bucket' returns discretized environment
@param state -> observation space 
"""
def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

#--------------------------------------------------------#

if __name__ == "__main__":
    print('Training ...')
    train()
    print('Testing ...')
    test()

#--------------------------------------------------------#
