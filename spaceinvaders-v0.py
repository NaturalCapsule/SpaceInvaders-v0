# Importing libraries
import gym # Make sure the gym version is 0.25.2, and you need to download this libraries 'pip install "gym[atari]" '
import numpy as np
from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

# making the environment
env = gym.make("SpaceInvaders-v0")

# Getting Observations
height, width, channels = env.observation_space.shape

# getting number of actions
actions = env.action_space.n

# Building the model using CNN
def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (8, 8), strides = (4, 4),
                     activation = 'relu', input_shape = (3, height, width, channels)))

    model.add(Conv2D(filters = 64, kernel_size = (4, 4), strides = (2, 2),
                     activation = 'relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                     activation = 'relu'))

    model.add(Flatten())

    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))

    model.add(Dense(actions, activation = 'linear'))

    return model

cnn = build_model(height, width, channels, actions)

# Building the agent
def build_agent():
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr = 'eps', value_max = 1.,
                                  value_min = .1, value_test = .2, nb_steps = 10000)
    memory = SequentialMemory(limit = 1000, window_length = 3)
    dqn = DQNAgent(model = cnn, memory = memory, policy = policy,
                   enable_dueling_network = True, dueling_type = 'avg',
                   nb_actions = actions, nb_steps_warmup = 10000)
    return dqn

agent = build_agent()

# Compiling the model
agent.compile(Adam(3e-4), metrics = ['mae'])

# Fitting the model
agent.fit(env, nb_steps = 10000, verbose = 2)

# Visualizing the agent and getting the score
results = agent.test(env, nb_episodes = 10, visualize = True)
print(np.mean(results.history['episode_reward']))

# Closing the environment
env.close()