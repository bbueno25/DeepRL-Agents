"""
An implementation of Arthur Juliani contextual bandits.
(https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c)
"""
from numpy import argmax as np_argmax
from numpy import argmin as np_argmin
from numpy import array as np_array
from numpy import mean as np_mean
from numpy import random as np_random
from numpy import zeros as np_zeros
from tensorflow import argmax as tf_argmax
from tensorflow import global_variables_initializer
from tensorflow import float32 as tf_float32
from tensorflow import int32 as tf_int32
from tensorflow import log as tf_log
from tensorflow import nn as tf_nn
from tensorflow import ones_initializer
from tensorflow import placeholder as tf_placeholder
from tensorflow import reset_default_graph
from tensorflow import reshape as tf_reshape
from tensorflow import Session
from tensorflow import slice as tf_slice
from tensorflow import train as tf_train
from tensorflow import trainable_variables
from tensorflow.contrib import slim as tf_slim


class Agent():

    def __init__(self, a_size, s_size, learning_rate=0.001):
        self.state_in = tf_placeholder(tf_int32, [1])
        output = tf_slim.fully_connected(tf_slim.one_hot_encoding(self.state_in, s_size),
                                         a_size,
                                         activation_fn=tf_nn.sigmoid,
                                         biases_initializer=None,
                                         weights_initializer=ones_initializer())
        output = tf_reshape(output, [-1])
        self.chosen_action = tf_argmax(output, 0)
        self.reward_holder = tf_placeholder(tf_float32, [1])
        self.action_holder = tf_placeholder(tf_int32, [1])
        self.responsible_weight = tf_slice(output, self.action_holder, [1])
        loss = -tf_log(self.responsible_weight) * self.reward_holder
        optimizer = tf_train.GradientDescentOptimizer(learning_rate)
        self.update = optimizer.minimize(loss)


class ContextualBandits(object):

    def __init__(self, bandits):
        self.bandits = bandits
        self.num_actions = self.bandits.shape[1]
        self.num_bandits = self.bandits.shape[0]
        self.state = 0

    def get_bandit(self):
        """ Returns a random state for each episode. """
        return np_random.randint(0, len(self.bandits))                                             # pylint: disable=E1101

    def pull_arm(self, action):
        bandit = self.bandits[self.state, action]
        if np_random.randn(1) > bandit:                                                            # pylint: disable=E1101
            return 1
        return -1


def main(e=0.1, num_episodes=10000):
    reset_default_graph()
    contextual_bandits = ContextualBandits(np_array([[0.2, 0.0, -0.0, -5.],
                                                     [0.1, -5.0, 1.0, 0.25],
                                                     [-5.0, 5.0, 5.0, 5.0]]))

    agent = Agent(contextual_bandits.num_actions, contextual_bandits.num_bandits)
    total_reward = np_zeros([contextual_bandits.num_bandits, contextual_bandits.num_actions])
    weights = trainable_variables()[0]

    with Session() as sess:
        sess.run(global_variables_initializer())

        for i in range(num_episodes):
            bandit = contextual_bandits.get_bandit()

            if np_random.rand(1) < e:                                                               # pylint: disable=E1101
                action = np_random.randint(contextual_bandits.num_actions)                          # pylint: disable=E1101
            else:
                action = sess.run(agent.chosen_action, {agent.state_in:[bandit]})

            reward = contextual_bandits.pull_arm(action)
            fetches = [agent.update, weights]
            feed_dict = {agent.reward_holder: [reward],
                         agent.action_holder: [action],
                         agent.state_in: [bandit]}
            _, ww = sess.run(fetches, feed_dict)
            total_reward[bandit, action] += reward

            if i % 500 == 0:
                print("Mean reward for each of the " + str(contextual_bandits.num_bandits) +
                      " bandits: " + str(np_mean(total_reward, axis=1)))

    for a in range(contextual_bandits.num_bandits):
        print("The agent thinks action " + str(np_argmax(ww[a]) + 1) +
              " for contextual_bandit " + str(a + 1) + " is the most promising.")
        if np_argmax(ww[a]) == np_argmin(contextual_bandits.bandits[a]):
            print("It was right!")
        else:
            print("It was wrong!")


if __name__ == "__main__":
    main()
