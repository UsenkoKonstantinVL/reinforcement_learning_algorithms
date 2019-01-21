import gym
import tensorflow as tf
import numpy as np
import random
import math


MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 64

GYME_NAME = 'CartPole-v0'
EPOCH = 20000
EPISODE = 180
EPOCH_PER_TRAINING = 5

RENDER = False


config = tf.ConfigProto(device_count={'GPU': 0})


class Agent:

    render = True

    def __init__(self,  num_states, num_actions, batch_size):
        # Define states, actions and batch size
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # Define states and actions tensor
        self._states = None
        self._actions = None
        # Define outputs and optimizer
        self._logits = None
        self._softmax_output = None
        self._optimizer = None
        self._var_init = None
        # Setup the model
        self.learning_rate = 0.0005
        self.build_model()
        self.discount_factor = 0.99
        self.model = self.build_model()
        self.states, self.actions, self.rewards = [], [], []

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float64)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float64)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 16, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 8, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions, activation=tf.nn.sigmoid)
        self._softmax_output = tf.layers.dense(self._logits, self._num_actions, activation=tf.nn.softmax)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._softmax_output, feed_dict={self._states: state.reshape(1, self.num_states)})

    # Return argument of action
    def predict_action(self, state, sess):
        policy = self.predict_one(state, sess)
        return np.random.choice(self._num_actions, 1, p=policy[0])[0]

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self._num_states))
        advantages = np.zeros((episode_length, self._num_actions))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        sess.run(self._optimizer, feed_dict={self._states: update_inputs, self._q_s_a: advantages})
        self.states, self.actions, self.rewards = [], [], []

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return 0

    @property
    def var_init(self):
        return self._var_init


def normalise_state(state):
    global max_state
    state[0] = state[0] / 1.6  # max_state[0]
    state[1] = state[1] / 1.5  # max_state[1]
    state[2] = state[2] / 12  # max_state[2]
    state[3] = state[3] / 0.6  # max_state[3]
    return state


env = gym.make(GYME_NAME)

num_states = env.env.observation_space.shape[0]
max_state = env.env.observation_space.high
num_actions = env.env.action_space.n

agent = Agent(num_states, num_actions, BATCH_SIZE)

stat_states = []

eps = MAX_EPSILON
with tf.Session(config=config) as sess:
    sess.run(agent.var_init)
    for epoch in range(EPOCH):
        _state = env.reset()
        state = normalise_state(_state)
        tot_reward = 0
        cum_reward = 0
        for episode in range(EPISODE):
            if RENDER:
                env.render()
            action = agent.predict_action(state, sess)
            _next_state, reward, done, info = env.step(action)
            next_state = normalise_state(_next_state)

            angle = -math.fabs(_next_state[2])
            angle_vel = -math.fabs(_next_state[3])

            cum_reward = reward  # angle + angle_vel
            tot_reward += cum_reward

            if done:
                cum_reward += -10
                tot_reward += -10
            elif (episode + 1) == EPISODE:
                cum_reward += 10
                tot_reward += 10

            agent.append_sample(state, action, cum_reward)

            stat_states.append(_next_state)

            if done:
                break
        agent.train_model()

        print("{}: sum reward: {}, episodes: {}".format(epoch + 1, tot_reward, episode))
        np_stat_states = np.array(stat_states)
        #print(np_stat_states.mean(0))
        print(np_stat_states.max(0))
