''' Once we are done installing gym we can load the environment and render it
The core gym interface is env, which is the unified environment interface. The following 
are the env methods that we will be using in this project.
env.reset: Resets the environment and returns a random initial state.
env.step(action): Step the environment by one timestep. Returns
observation: Observations of the environment
reward: If your action was beneficial or not
done: Indicates if we have successfully picked up and dropped off a passenger, 
also called one episode
info: Additional info such as performance and latency for debugging purposes

env.render: Renders one frame of the environment (helpful in visualizing the environment)
'''

import gym

env = gym.make("Taxi-v3").env # loads the invironment

env.render() #renders the environment

env.reset() #reset the environment to a new, random state
env.render()

print(f"Action Space {env.action_space}")
print(f"State Space {env.observation_space}")

'''The filled square represents the taxi, which is yellow without a passenger and
green with a passenger.The pipe ("|") represents a wall which the taxi cannot cross.
R, G, Y, B are the possible pickup and destination locations. 
The blue letter represents the current passenger pick-up location, and
the purple letter is the current destination.'''


state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.render()
'''When the environment was created, the initial reward table "P" also was created.
 We can think of it as a matrix of states x rows. '''


env.P[328]

# Qlearning Approach

import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])

%%time
"""Training the agent"""

import random
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done: # we decide whether to pick a random action or to exploit the already computed Q-values.
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

q_table[328]

#Evaluating agent's performance
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
