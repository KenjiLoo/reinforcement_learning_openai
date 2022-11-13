import gym
import random
import numpy as np
# import Plotter

policy = [[None for _ in range(0, 15)]for _ in range(0, 19)]  # Policy for agent defined by state space
q = [[[0 for _ in range(0, 3)]for _ in range(0, 15)]for _ in range(0, 19)]  # State-action-value matrix
# plotter = Plotter()
ep = 0  # Episode
max = 10000  # Max episode
latest_states = [[0 for _ in range(0, 3)]for _ in range(0, 1000)]  # States accessed in the latest episode
epsilon = 0  # Mutation rate
reward = []  # Episode reward matrix
mutated = []


def state_formatter(state):
    """
    formats state space to positive whole numbers
    """

    # Round state space to tenths place to form finite values

    state[0] = state[0].round(1)
    state[1] = state[1].round(2)

    # Make numbers whole

    state[0] *= 10
    state[1] *= 100

    # Make numbers whole

    state[0] += 12
    state[1] += 7

    # Change data type

    integer = [0,0]
    integer[0] = int(state[0])
    integer[1] = int(state[1])

    return integer


def calc_q(latest_states):
    """
    updates state values
    """
    for i in range(0,19):
        for p in range(0,15):
            for u in range(1000):

                # If value occurs in last episode, update it according to episode reward

                if [i, p, policy[i][p]] == [latest_states[u][0], latest_states[u][1], latest_states[u][2]]:
                    q[i][p][latest_states[u][2]] = (q[i][p][latest_states[u][2]]/2) + reward[ep]

    return q


def greedy():
    """
    calculates optimal policy
    """
    for i in range(0,19):
        for p in range(0,15):

            # If state is unexplored, do not calculate optimal action
            # Else, update the policy accoring to epsilon

            if q[i][p][0] == 0 and q[i][p][1] == 0 and q[i][p][2] == 0:
                policy[i][p] = None
            else:
                if epsilon > random.uniform(0,1):
                    policy[i][p] = q[i][p].index(np.max(q[i][p]))
                else:
                    policy[i][p] = random.randint(0,2)
    return policy


def calc_reward():
    """
    calculates episode reward by finding the maximum X value
    """
    arr = []
    for i in range(1000):
        arr.append(latest_states[i][0])
    return np.max(arr)


env = gym.make('MountainCar-v0', render_mode='human') # Make environemnt

while ep < max:
    epsilon += 0.001 # Update epsilon for convergence
    count = 0 # Reset frame counter
    formatted_state = state_formatter(env.reset()) # Reset environment
    while count < 999:

        # If state is unexplored, use uniform random action

        if policy[formatted_state[0]][formatted_state[1]] == None:
            policy[formatted_state[0]][formatted_state[1]] = random.randint(0,2)

        # Step forward in environment with action and receive new state

        state, rew, end, info, _ = env.step(policy[formatted_state[0]][formatted_state[1]])

        # Set local state space

        latest_states[count][0] = formatted_state[0]
        latest_states[count][1] = formatted_state[1]
        latest_states[count][2] = policy[formatted_state[0]][formatted_state[1]]

        # Format new state space

        formatted_state = state_formatter(state)
        count += 1

    reward.append(calc_reward()) # Append reward matrix for graph
    print('#'*reward[ep]+" "+str(reward[ep])) # Graph
    q = calc_q(latest_states) # Calculate Q
    policy = greedy() # e-greedy function
    if ep % 100 == 0:

        # DEBUG:
        print("EPISODE "+str(ep))
        print("----------------------------")
        print()
        # for i in policy:
        #     print(bcolors.HEADER+str(i))
        # print(bcolors.ENDC)
    ep += 1