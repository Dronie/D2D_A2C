import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import trfl
import matplotlib.pyplot as plt

class ActorCriticNet:
    def __init__(self, name, state_size=2, action_size=2, actor_hidden_size=32, critic_hidden_size=32, ac_learning_rate=0.001,  
                 entropy_cost=0.01, normalise_entropy=True, lambda_=0., baseline_cost=1.):

                with tf.variable_scope(name):

                    # placeholders
                    self.name = name
                    self.input = tf.placeholder(tf.float32, [None, state_size], name='input')
                    self.action = tf.placeholder(tf.int32, [None, 1], name='action')
                    self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')
                    self.discount = tf.placeholder(tf.float32, [None, 1], name='discount')
                    self.bootstrap = tf.placeholder(tf.float32, [None], name='bootstrap')

                    # actor net
                    self.actor_hidden_1 = tf.contrib.layers.fully_connected(self.input, actor_hidden_size, activation_fn=tf.nn.relu)
                    self.actor_hidden_2 = tf.contrib.layers.fully_connected(self.actor_hidden_1, actor_hidden_size, activation_fn=tf.nn.relu)
                    self.actor_out = tf.contrib.layers.fully_connected(self.actor_hidden_2, action_size, activation_fn=None)

                    # policy logits
                    self.policy_logits = tf.reshape(self.actor_out, [-1, 1, action_size], name='policy_logits')

                    # action choice 
                    self.action_choice = tf.nn.softmax(self.actor_out)
                    
                    # critic net
                    self.critic_hidden_1 = tf.contrib.layers.fully_connected(self.input, critic_hidden_size, activation_fn=tf.nn.relu)
                    self.critic_hidden_2 = tf.contrib.layers.fully_connected(self.critic_hidden_1, critic_hidden_size, activation_fn=tf.nn.relu)
                    self.critic_out = tf.contrib.layers.fully_connected(self.critic_hidden_2, critic_hidden_size, activation_fn=None)

                    # loss function
                    self.acloss = trfl.sequence_advantage_actor_critic_loss(self.policy_logits, self.critic_out, self.action,
                                                                            self.reward, self.discount, self.bootstrap, lambda_=lambda_, entropy_cost=entropy_cost, 
                                                                            baseline_cost=baseline_cost, normalise_entropy=normalise_entropy)
                    
                    # optimizer
                    self.ac_optim = tf.reduce_mean(self.acloss.loss)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=ac_learning_rate).minimize(self.ac_optim)

    def get_network_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]