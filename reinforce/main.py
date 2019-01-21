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
EPISODE = 200
EPOCH_PER_TRAINING = 5

RENDER = True


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
        self._optimizer = None
        self._var_init = None
        # Setup the model
        self.build_model()

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float64)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float64)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 15, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 15, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions, activation=tf.nn.sigmoid)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self.num_states)})

    # Return argument of action
    def predict_action(self, state, sess):
        policy = self.predict_one(state, sess)
        return np.random.choice(self.action_size, 1, p=policy)

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

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        sess.run(self._optimizer, feed_dict={self._states: update_inputs, self._q_s_a: advantages})
        self.states, self.actions, self.rewards = [], [], []


def choose_action(sess, agent, state):
    global eps, MIN_EPSILON, MAX_EPSILON
    if random.random() < eps:
        act = random.randint(0, agent.num_actions - 1)
    else:
        act = np.argmax(agent.predict_one(state, sess))
    if eps > MIN_EPSILON:
        eps = eps - (MAX_EPSILON - MIN_EPSILON) / 6000
    return act


def normalise_state(state):
    global max_state
    state[0] = state[0] / max_state[0]
    state[1] = state[1] / max_state[1]
    state[2] = state[2] / max_state[2]
    state[3] = state[3] / max_state[3]
    return state


def replay(sess, agent, memory):
    batch = memory.sample(agent.batch_size)
    states = np.array([val[0] for val in batch])
    next_states = np.array([(np.zeros(agent.num_states)
                             if val[3] is None else val[3]) for val in batch])
    # predict Q(s,a) given the batch of states
    q_s_a = agent.predict_batch(states, sess)
    # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
    q_s_a_d = agent.predict_batch(next_states, sess)
    # setup training arrays
    x = np.zeros((len(batch), agent.num_states))
    y = np.zeros((len(batch), agent.num_actions))
    for i, b in enumerate(batch):
        state, action, reward, next_state = b[0], b[1], b[2], b[3]
        # get the current q values for all actions in state
        current_q = q_s_a[i]
        # update the q value for action
        if next_state is None:
            # in this case, the game completed after action, so there is no max Q(s',a')
            # prediction possible
            current_q[action] = reward
        else:
            current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
        x[i] = state
        y[i] = current_q
    agent.train_batch(sess, x, y)


env = gym.make(GYME_NAME)

num_states = env.env.observation_space.shape[0]
max_state = env.env.observation_space.high
num_actions = env.env.action_space.n

agent = Agent(num_states, num_actions, BATCH_SIZE)
# TODO: delete mem
mem = Memory(50000)

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
            action = choose_action(sess, agent, state)
            _next_state, reward, done, info = env.step(action)
            next_state = normalise_state(_next_state)
            cum_reward += reward
            if done:
                cum_reward = -10
            tot_reward += cum_reward
            agent.append_sample()
            mem.add_sample((state, action, cum_reward, next_state))

            # Train our network
            if (epoch + 1) % EPOCH_PER_TRAINING == 0:
                replay(sess, agent, mem)

            if done:
                break

        print("{}: sum reward: {}".format(epoch + 1, tot_reward))
