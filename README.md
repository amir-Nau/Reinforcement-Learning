# Reinforced-Learning

Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Safe state abstraction in reinforcement learning allows an
agent to ignore aspects of its current state that are irrelevant to its current decision, and therefore speeds up dynamic programming and learning. In this repository we will explore our understanding of RL.

In this repository I will try to design a simulation of a self-driving cab. The major goal is demonstrate how reinforced learning tackles these sorts of problems. 

Our smart cab ultimately should be able to do the following task: 
1.Passenger drop off to the right location
2.Take minimum possible time to drop off the passenger
3.Follow traffic rules

    The aspects I will be considering while designing this model are Rewards, and Actions.
    
## Rewards

Since our cab is motivated by rewards and is going to control the cab by trial experiences in the environment, we need to decide the rewards and/or penalties and their magnitude. The agent will recieve a high positive reward for a siccessful drop off since this behavior is highly desirable. On the other hand, the agent will be penalized if it tries to drop off in wrong location. Finallly, the agent will recieve slight negative reward at each time-space for not making it to destination. We give a "slight" negative reward in order for our agent to reach the destination even if it's late rather than making wrong moves to try to reach the destination as soon as possible.

## State Space

    In Reinforcement Learning, the agent encounters a state, and then takes action according to the state it's in.
The State Space is the set of all possible situations our cab will encounter. 

## Action Space

We have six possible actions in total. 
These are:

1. South
2. North
3. East
4. West 
5.Pickup
6. Dropoff

We will use the Gym environment called Taxi-v3. To use that run the following command on terminal or jupyter notebook. 
    !pip install cmake 'gym[atari]' scipy
    

