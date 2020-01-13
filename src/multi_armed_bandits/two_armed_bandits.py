"""
An implementation of Arthur Juliani two-armed bandits.
(https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149)
"""
from numpy import argmax as np_argmax
from numpy import array as np_array
from numpy import random as np_random
from numpy import zeros as np_zeros
from tensorflow import argmax as tf_argmax
from tensorflow import global_variables_initializer
from tensorflow import float32 as tf_float32
from tensorflow import int32 as tf_int32
from tensorflow import log as tf_log
from tensorflow import ones as tf_ones
from tensorflow import placeholder as tf_placeholder
from tensorflow import reset_default_graph
from tensorflow import Session
from tensorflow import slice as tf_slice
from tensorflow import train as tf_train
from tensorflow import Variable as tf_Variable


class Agent(object):

    def __init__(self, num_bandits, learning_rate=0.001):
        self.weights = tf_Variable(tf_ones([num_bandits]))
        self.chosen_action = tf_argmax(self.weights, 0)
        self.action_holder = tf_placeholder(tf_int32, [1])
        self.reward_holder = tf_placeholder(tf_float32, [1])
        self.responsible_weight = tf_slice(self.weights, self.action_holder, [1])
        loss = -(tf_log(self.responsible_weight) * self.reward_holder)
        optimizer = tf_train.GradientDescentOptimizer(learning_rate)
        self.update = optimizer.minimize(loss)


class TwoArmedBandits(object):

    def __init__(self, bandits):
        self.bandits = bandits
        self.num_bandits = len(self.bandits)

    def pull_arm(self, bandit):
        if np_random.randn(1) > bandit:                                                            # pylint:disable=E1101
            return 1
        return -1


def main(e=0.1, num_episodes=1000):
    reset_default_graph()
    two_armed_bandits = TwoArmedBandits([0.2, 0.0, -0.2, -5.0])
    agent = Agent(two_armed_bandits.num_bandits)
    total_reward = np_zeros(two_armed_bandits.num_bandits)

    with Session() as sess:
        sess.run(global_variables_initializer())

        for i in range(num_episodes):
            if np_random.rand(1) < e:                                                              # pylint:disable=E1101
                action = np_random.randint(two_armed_bandits.num_bandits)                          # pylint:disable=E1101
            else:
                action = sess.run(agent.chosen_action)

            reward = two_armed_bandits.pull_arm(two_armed_bandits.bandits[action])
            fetches = [agent.update, agent.responsible_weight, agent.weights]
            feed_dict = {agent.reward_holder: [reward], agent.action_holder: [action]}
            _, _, ww = sess.run(fetches, feed_dict)
            total_reward[action] += reward

            if i % 50 == 0:
                print("Number of bandits: ", two_armed_bandits.num_bandits)
                print("Running reward for the ", total_reward)

    promising = np_argmax(ww) + 1
    print("The agent thinks bandit " + str(promising) + " is the most promising.")

    if np_argmax(ww) == -(np_argmax(np_array(two_armed_bandits.bandits))):
        print("It was right!")
    else:
        print("It was wrong!")


if __name__ == "__main__":
    main()
