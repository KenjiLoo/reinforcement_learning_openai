import numpy as np
import random
import collections
from collections import deque
import random

# for building the DQN model
from keras import layers
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation
from keras.layers import LeakyReLU
from keras.initializers import he_normal, RandomNormal
from keras.optimizers import SGD


class MountainCarAgent():
    """
    The playing agent.
    """

    def __init__(self, action_size, state_size, discount_factor=0.95, learning_rate=0.02,
                 epsilon=1, epsilon_decay=0.99, epsilon_min=0.02):

        # parameters
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_max = 1

        self.load_model = True
        self.epsilon_min = epsilon_min
        self.batch_size = 32
        self.memory = deque(maxlen=2000)

        # action and state sizes
        self.action_size = action_size
        self.state_size = state_size

        # build the NN model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    #         if self.load_model:
    #             self.model.load_weights("./model_weights.h5")

    def build_model(self):

        initializer = he_normal(seed=1000)
        model = Sequential()
        model.add(Dense(48, input_dim=2, kernel_initializer=initializer))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dense(64, kernel_initializer=initializer))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dense(self.action_size, activation=LeakyReLU(alpha=0.1), kernel_initializer=initializer))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=0.0001))

        return model

    def get_action(self, state):
        """
        get action in a state according to an epsilon-greedy approach
        """

        if random.uniform(0, 1) <= self.epsilon:
            # explore: choose a random action from all possible actions
            # in case of MountainCar this will randomly choose an action between 0, 1 and 2
            #             print('random')
            return random.randrange(self.action_size)
        else:
            # choose the action with the highest q(s, a)
            # the first index corresponds to the batch size, so
            # reshape state to (1, state_size) so that the first index corresponds to the batch size
            state = state.reshape(1, self.state_size)
            q_value = self.model.predict(state)

            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        # append the tuple (s, a, r, s', done) to memory (replay buffer) after every action
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        '''
        TODO:
        Update the target Q-value network to current Q-value network after training for a episode. This means that weights an
        biases of target Q-value network will become same as current Q-value network.
        '''
        self.target_model.set_weights(self.model.get_weights())

    def train_model(self):
        """
        train the neural network on a minibatch. Input to the network is the states,
        output is the target q-value corresponding to each action.
        """
        if len(self.memory) > self.batch_size:

            # sample minibatch from memory
            minibatch = random.sample(self.memory, self.batch_size)
            #             print('--minibatch')
            #             print(np.array(minibatch).shape)
            # initialise two matrices - update_input and update_output
            update_input = np.zeros((self.batch_size, self.state_size))
            update_output = np.zeros((self.batch_size, self.state_size))
            actions, rewards, done = [], [], []

            # populate update_input and update_output and the lists rewards, actions, done
            for i in range(self.batch_size):
                state, action, reward, next_state, done_boolean = minibatch[i]
                #                 print('mini batch')
                #                 print(minibatch[i])
                #                 print(state)
                update_input[i] = state
                actions.append(action)
                rewards.append(reward)
                update_output[i] = next_state
                done.append(done_boolean)

            # predict the target q-values from states s
            target = self.model.predict(update_input)

            # target for q-network
            target_qval = self.target_model.predict(update_output)
            #             print('---update input')
            #             print(update_input.shape)
            #             print('---update output')
            #             print(update_output.shape)
            #             print('---target')
            #             print(target.shape)
            #             print('---target qvalue')
            #             print(target_qval.shape)
            #             print('--target qval shape')
            #             print(target_qval.shape)
            #             print(np.max)

            # update the target values
            for i in range(self.batch_size):
                if done[i]:
                    #                     print('--target qval')
                    #                     print(target_qval[i])
                    #                     print(np.max(target_qval[i]))
                    #                     print('---action')
                    #                     print(actions)
                    #                     print(len(actions))
                    #                     print(actions[i])

                    target[i][actions[i]] = rewards[i]
                else:  # non-terminal state
                    target[i][actions[i]] = rewards[i] + self.discount_factor * np.max(target_qval[i])

            # model fit
            self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0, validation_split=0.1)

    def save_model_weights(self, name):
        self.model.save_weights(name)
