
# This code was adapted from work done by GitHub user 'afzal63'
# a link to the specific peice of work used is listed below
# (https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-TensorFlow-TRFL/blob/master/Section%203/Actor-Critic.ipynb)

import tensorflow as tf
import sys
import numpy as np
import tensorflow_probability as tfp
import trfl
import matplotlib.pyplot as plt
import D2D_env_discrete as D2D
import json

# instantiate the simulator
ch = D2D.Channel()

# set up Actor and Critic networks
class ActorCriticNetwork:
    def __init__(self, name, obs_size=2, action_size=2, actor_hidden_size=32, critic_hidden_size=32, pow_learning_rate=0.001,  
                   RB_learning_rate=0.001, entropy_cost=0.01, normalise_entropy=True, lambda_=0., baseline_cost=1.):
    
        with tf.variable_scope(name):
            
            # define inputs placeholders for the networks
            self.name=name
            self.input_ = tf.placeholder(tf.float32, [None, obs_size], name='inputs')
            self.action_RB_ = tf.placeholder(tf.int32, [None, 1], name='action_RB')
            self.action_pow_ = tf.placeholder(tf.int32, [None, 1], name='action_pow')
            self.reward_ = tf.placeholder(tf.float32, [None, 1], name='reward')
            self.discount_ = tf.placeholder(tf.float32, [None, 1], name='discount')
            self.bootstrap_ = tf.placeholder(tf.float32, [None], name='bootstrap')

            # set up actor network
            self.fc1_actor_ = tf.contrib.layers.fully_connected(self.input_, actor_hidden_size, activation_fn=tf.nn.elu)
            self.fc2_actor_ = tf.contrib.layers.fully_connected(self.fc1_actor_, actor_hidden_size, activation_fn=tf.nn.elu)
            self.fc3_actor_ = tf.contrib.layers.fully_connected(self.fc2_actor_, actor_hidden_size, activation_fn=tf.nn.elu)

            self.fc4_actor_power_ = tf.contrib.layers.fully_connected(self.fc3_actor_, ch.D2D_tr_Power_levels, activation_fn=None)            
            self.fc4_actor_RB_ = tf.contrib.layers.fully_connected(self.fc3_actor_, ch.N_CU, activation_fn=None)

            # reshape the policy logits
            self.policy_logits_RB_ = tf.reshape(self.fc4_actor_RB_, (-1, 1, ch.N_CU))
            self.policy_logits_power_ = tf.reshape(self.fc4_actor_power_, (-1, 1, ch.D2D_tr_Power_levels))
             
            # generate action probabilities for taking actions
            self.action_prob_power_ = tf.nn.softmax(self.fc4_actor_power_)
            self.action_prob_RB_ = tf.nn.softmax(self.fc4_actor_RB_)
      
            # set up critic network
            self.fc1_critic_ = tf.contrib.layers.fully_connected(self.input_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc2_critic_ = tf.contrib.layers.fully_connected(self.fc1_critic_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.baseline_ = tf.contrib.layers.fully_connected(self.fc2_critic_, 1, activation_fn=None)
      
            # Define Loss with TRFL
            self.seq_aac_return_pow_ = trfl.sequence_advantage_actor_critic_loss(self.policy_logits_power_, self.baseline_, self.action_pow_,
                self.reward_, self.discount_, self.bootstrap_, lambda_=lambda_, entropy_cost=entropy_cost, 
                baseline_cost=baseline_cost, normalise_entropy=normalise_entropy)          

            self.seq_aac_return_RB_ = trfl.sequence_advantage_actor_critic_loss(self.policy_logits_RB_, self.baseline_, self.action_RB_,
                self.reward_, self.discount_, self.bootstrap_, lambda_=lambda_, entropy_cost=entropy_cost, 
                baseline_cost=baseline_cost, normalise_entropy=normalise_entropy)

            # Optimize the loss
            self.ac_loss_pow_ = tf.reduce_mean(self.seq_aac_return_pow_.loss)
            self.ac_loss_RB_ = tf.reduce_mean(self.seq_aac_return_RB_.loss)
            self.ac_optim_pow_ = tf.train.AdamOptimizer(learning_rate=pow_learning_rate).minimize(self.ac_loss_pow_)
            self.ac_optim_RB_ = tf.train.AdamOptimizer(learning_rate=RB_learning_rate).minimize(self.ac_loss_RB_)
            
    # used to pass parameters to target networks
    def get_network_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]

# hyperparameters
train_episodes = 5000  
discount = 0.99

actor_hidden_size = 64
critic_hidden_size = 64
    
pow_learning_rate = 0.001
RB_learning_rate = 0.001

target_pow_learning_rate = 0.001
target_RB_learning_rate = 0.001

baseline_cost = 10 #scale derivatives between actor and critic networks

entropy_cost = 0.001 
normalise_entropy = True

lambda_ = 1. # not used

# get action and state sizes respectively
action_size = ch.n_actions
obs_size = ch.N_CU

# reset tensorflow graphs
tf.reset_default_graph()

# instantiate actor and critic networks
D2D_nets = []
D2D_target_nets = []
D2D_target_net_update_ops = []

for i in range(0, ch.N_D2D):
    D2D_nets.append(ActorCriticNetwork(name='ac_net_{:.0f}'.format(i), obs_size=obs_size, action_size=action_size, actor_hidden_size=actor_hidden_size,
                                       pow_learning_rate=pow_learning_rate, RB_learning_rate=RB_learning_rate, entropy_cost=entropy_cost, normalise_entropy=normalise_entropy,
                                       lambda_=lambda_, baseline_cost=baseline_cost))

    print('Instantiated Network {:.0f} of {:.0f}'.format(i+1, ch.N_D2D))

    D2D_target_nets.append(ActorCriticNetwork(name='ac_target_net_{:.0f}'.format(i), obs_size=obs_size, action_size=action_size, actor_hidden_size=actor_hidden_size,
                                       pow_learning_rate=target_pow_learning_rate, RB_learning_rate=target_RB_learning_rate, entropy_cost=entropy_cost, normalise_entropy=normalise_entropy,
                                       lambda_=lambda_, baseline_cost=baseline_cost))

    print('Instantiated Target Network {:.0f} of {:.0f}'.format(i+1, ch.N_D2D))

    D2D_target_net_update_ops.append(trfl.update_target_variables(D2D_target_nets[i].get_network_variables(), 
                                                                  D2D_nets[i].get_network_variables(), tau=0.001))

    print('Instantiated Target Net Update ops {:.0f} of {:.0f}'.format(i+1, ch.N_D2D))
    print('\n')

# used for progress reports in console
stats_rewards_list = []
stats_every = 10

# initialize the simulator and get channel gains
g_iB, g_j, G_ij, g_jB, G_j_j, d_ij = ch.reset()

# set initial state to list of 0s (one for each Cellular User
state = np.zeros(ch.N_CU)

# used in plot smoothing procedure
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

# start tensorflow session
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    total_reward, ep_length, done = 0, 0, 0
    stats_actor_loss_pow, stats_critic_loss_pow = 0., 0.
    stats_actor_loss_RB, stats_critic_loss_RB = 0., 0.
    total_loss_list_pow, total_loss_list_RB, action_list, action_prob_list, bootstrap_list = [], [], [], [], []
    rewards_list = []
    collision_var = 0
    D2D_collision_probs = []
    collisions = []
    access_ratios = []
    access_rates = []
    avg_throughput = []
    time_avg_throughput = []

    # begin training
    for ep in range(1, train_episodes):
        
        # used to determine the collision probability
        ch.collision_indicator = 0
                     
        # generate action probabilities from policy net and sample from the action probs
        power_action_probs = []
        RB_action_probs = []
        pow_sels = []
        RB_sels = []

        for i in range(0, ch.N_D2D):
            # get power level selection action probabilities from respective head in actor network for each agent
            power_action_probs.append(sess.run(D2D_nets[i].action_prob_power_, feed_dict={D2D_nets[i].input_: np.expand_dims(state,axis=0)}))
            power_action_probs[i] = power_action_probs[i][0]

            # get RB selection action probabilities from respective head in actor network for each agent
            RB_action_probs.append(sess.run(D2D_nets[i].action_prob_RB_, feed_dict={D2D_nets[i].input_: np.expand_dims(state,axis=0)}))
            RB_action_probs[i] = RB_action_probs[i][0]

            # make a random choice of action based on the action probabilities
            pow_sels.append(np.random.choice(ch.power_levels, p=power_action_probs[i]))
            RB_sels.append(np.random.choice(ch.CU_index, p=RB_action_probs[i]))

        # get cellular user SINRs based on channel gains and action selections
        CU_SINR = ch.CU_SINR_no_collision(g_iB, pow_sels, g_jB, RB_sels)

        # get the next state from the simulator
        next_state = ch.state(CU_SINR)

        # get the D2D SINR based on the next state, channel gains and action selections
        D2D_SINR = ch.D2D_SINR_no_collision(pow_sels, g_j, G_ij, G_j_j, RB_sels, next_state)

        # get rewards from simulator
        reward, net, _, _ = ch.D2D_reward_no_collision(D2D_SINR, CU_SINR, RB_sels, d_ij)
        
        # divide rewards by channel bandwidth (to simplify learning)
        reward = reward / 10**10
        net = net / 10**10
        
        # used for determining the collision probability
        if ch.collision_indicator > 0:
            collision_var += 1
        collisions.append(ch.collision_indicator)
        D2D_collision_probs.append(collision_var / ep)

        # prepare the next state for input to networks
        next_state = np.clip(next_state,-1.,1.)
        # update the total reward counter
        total_reward += net

        # used for post-training plots
        ep_length += 1

        # get bootstrap values from target nets
        if ep == train_episodes:
            # placeholder bootstrap value for the terminal state
            bootstrap_value = np.zeros((1,),dtype=np.float32)
        else:
            #get bootstrap value
            bootstrap_values = []
            for i in range(0, ch.N_D2D):
                bootstrap_values.append(sess.run(D2D_target_nets[i].baseline_, feed_dict={
                    D2D_target_nets[i].input_: np.expand_dims(next_state, axis=0)}))
        
        # used for post training plots
        total_losses_pow = []
        seq_aac_returns_pow = []
        total_losses_RB = []
        seq_aac_returns_RB = []

         # network updates
        for i in range(0, ch.N_D2D):
            # update networks through actor power level selection destribution head 
            _, total_loss_pow, seq_aac_return_pow = sess.run([D2D_nets[i].ac_optim_pow_, D2D_nets[i].ac_loss_pow_, D2D_nets[i].seq_aac_return_pow_], feed_dict={
                D2D_nets[i].input_: np.expand_dims(state, axis=0),
                D2D_nets[i].action_pow_: np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
                D2D_nets[i].action_RB_: np.reshape(RB_sels[i], (-1, 1)),
                D2D_nets[i].reward_: np.reshape(reward[i], (-1, 1)),
                D2D_nets[i].discount_: np.reshape(discount, (-1, 1)),
                D2D_nets[i].bootstrap_: np.reshape(bootstrap_values[i], (1,))
            })

            # used for post training plots
            total_losses_pow.append(total_loss_pow)
            seq_aac_returns_pow.append(seq_aac_return_pow)

        for i in range(0, ch.N_D2D):
            # update networks through actor RB selection destribution head 
            _, total_loss_RB, seq_aac_return_RB = sess.run([D2D_nets[i].ac_optim_RB_, D2D_nets[i].ac_loss_RB_, D2D_nets[i].seq_aac_return_RB_], feed_dict={
                D2D_nets[i].input_: np.expand_dims(state, axis=0),
                D2D_nets[i].action_pow_: np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
                D2D_nets[i].action_RB_: np.reshape(RB_sels[i], (-1, 1)),
                D2D_nets[i].reward_: np.reshape(reward[i], (-1, 1)),
                D2D_nets[i].discount_: np.reshape(discount, (-1, 1)),
                D2D_nets[i].bootstrap_: np.reshape(bootstrap_values[i], (1,))
            })

            # used for post training plots
            total_losses_RB.append(total_loss_RB)
            seq_aac_returns_RB.append(seq_aac_return_RB)
        total_loss_list_pow.append(np.mean(total_losses_pow))
        total_loss_list_RB.append(np.mean(total_losses_RB))
        
        #update target network
        for i in range(0, ch.N_D2D):
            sess.run(D2D_target_net_update_ops[i])
        
        # update state
        state = next_state

        # ALL FOLLOWING CODE IS FOR DEBUGGING / POST-TRAINING PLOTS
        a = list(ch.accessed_CUs)
        if 2 in a:
            a = a.index(2)
            b = RB_sels.index(a)
        else:
            b = 0
        
        accessed = []
        throughput = []    
        for i in range(0, len(reward)):
            if reward[i] > 0:
                accessed.append(reward[i])
                throughput.append(reward[i])
            else:
                throughput.append(0.0)
        
        access_ratios.append(len(accessed) / len(reward))
        access_rates.append(sum(access_ratios) / ep)

        avg_throughput.append(sum(throughput))
        time_avg_throughput.append(sum(avg_throughput)/ ep)

        if ep % stats_every == 0 or ep == 1:
            print('Power Levels: ', pow_sels)
            print('RB Selections: ', RB_sels)
            print('Accessed CUs: ', ch.accessed_CUs)
            print('Rewards of agents: ', reward)
            print('Number of Collisions: ', ch.collision_indicator)
            print('||(Ep)isode: {}|| '.format(ep),
                  'Last net (r)eward: {:.3f}| '.format(net),
                  'Throughput: {:.3f}| '.format(sum(throughput)),
                  'Pow (L)oss: {:4f}|'.format(np.mean(total_loss_list_pow)),
                  'RB (L)oss: {:4f}|'.format(np.mean(total_loss_list_RB)))

        stats_rewards_list.append((ep, total_reward, ep_length))
        rewards_list.append(net)
        

    eps, rews, lens = np.array(stats_rewards_list).T
    smoothed_rews = running_mean(rewards_list, 100)
    smoothed_col_probs = running_mean(D2D_collision_probs, 100)
    smoothed_access_rates = running_mean(access_rates, 100)

    smoothed_throughput = running_mean(time_avg_throughput, 100)

    sr = open('indTwoOutRew.json', 'w+')
    json.dump(list(smoothed_rews), sr)

    sc = open('indTwoOutCol.json', 'w+')
    json.dump(list(smoothed_col_probs), sc)

    sa = open('indTwoOutAcc.json', 'w+')
    json.dump(list(smoothed_access_rates), sa)
    
    st = open('indTwoOutThr.json', 'w+')
    json.dump(list(smoothed_throughput), st)

    reward_fig = plt.figure()

    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rewards_list, color='grey', alpha=0.3)
    plt.xlabel('Time-slot')
    plt.ylabel('Reward')
    plt.show()

    collision_prob_fig = plt.figure()
    plt.ylim(0, 1)
    plt.xlim(0, 5000)
    plt.plot(eps[-len(smoothed_col_probs):], smoothed_col_probs)
    plt.plot(eps, D2D_collision_probs, color='grey', alpha=0.3)
    plt.xlabel('Time-slot')
    plt.ylabel('D2D collision probability')
    plt.show()

    true_collisions_fig = plt.figure()
    plt.ylim(0, 1)
    plt.xlim(0, 5000)
    plt.plot(eps, collisions)
    plt.ylabel('Number of collisions')
    plt.xlabel('Time-slot')
    plt.show()

    access_rate_fig = plt.figure()
    plt.ylim(0, 1)
    plt.xlim(0, 5000)
    plt.plot(eps[-len(smoothed_access_rates):], smoothed_access_rates)
    plt.plot(eps, access_rates, color='grey', alpha=0.3)
    plt.xlabel('Time-slot')
    plt.ylabel('D2D access rate')
    plt.show()

    time_avg_overall_thrghpt_fig = plt.figure()
    plt.plot(eps[-len(smoothed_throughput):], smoothed_throughput)
    plt.plot(eps, time_avg_throughput, color='grey', alpha=0.3)
    plt.xlabel('Time-slot')
    plt.ylabel('Time-averaged network throughput')
    plt.show()
    

