
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
import statistics as stats
import collections
import json

# instantiate the simulator
ch = D2D.Channel()

# Actor Network Class
class ActorNetwork:
    def __init__(self, name, obs_size=2, actor_hidden_size=32, actor_learning_rate=0.001):
    
        with tf.variable_scope(name, "Actor"):
            
            # define inputs placeholders for the actor network
            self.name=name
            self.input_ = tf.placeholder(tf.float32, [None, obs_size], name='inputs')
            self.action_ = tf.placeholder(tf.int32, [None, 1], name='action_RB')
            self.action_values_ = tf.placeholder(tf.float32, [None, 1], name='action_values')

            # set up actor network
            self.fc1_actor_ = tf.contrib.layers.fully_connected(self.input_, actor_hidden_size, activation_fn=tf.nn.elu)
            self.fc2_actor_ = tf.contrib.layers.fully_connected(self.fc1_actor_, actor_hidden_size, activation_fn=tf.nn.elu)
            self.fc3_actor_ = tf.contrib.layers.fully_connected(self.fc2_actor_, actor_hidden_size, activation_fn=tf.nn.elu)
            self.fc4_actor_ = tf.contrib.layers.fully_connected(self.fc3_actor_, actor_hidden_size, activation_fn=tf.nn.elu)
            self.fc5_actor = tf.contrib.layers.fully_connected(self.fc4_actor_, ch.n_actions, activation_fn=None)

            # reshape the policy logits
            self.policy_logits = tf.reshape(self.fc5_actor, (-1, 1, ch.n_actions))
             
            # generate action probabilities for taking actions
            self.action_prob = tf.nn.softmax(self.fc5_actor)
            
            # get actor loss
            self.Actor_return = trfl.discrete_policy_gradient(self.policy_logits, self.action_, self.action_values_)

            # Optimize the loss
            self.ac_loss = tf.reduce_mean(self.Actor_return)
            self.ac_optim = tf.train.AdamOptimizer(learning_rate=actor_learning_rate).minimize(self.ac_loss)

# COMA critic network
class CriticNetwork:
    def __init__(self, name, obs_size=None, action_size=None, critic_hidden_size=32, critic_learning_rate = 0.0001):
        with tf.variable_scope(name, "Critic"):
            
            # define inputs for COMA critic network
            self.name=name
            self.state_ = tf.placeholder(tf.float32, [None, obs_size], name='state')
            self.joint_action_min_a = tf.placeholder(tf.float32, [None, ch.N_D2D - 1], name='joint_actions_min_a')
            self.current_actor = tf.placeholder(tf.float32, [None, 1], name='current_actor')
            self.joint_action_tm1 = tf.placeholder(tf.float32, [None, ch.N_D2D], name='joint_action_tm1')
            self.actions_ = tf.placeholder(tf.int32, [None, 1], name='c_actions')
            self.reward_ = tf.placeholder(tf.float32, [None, 1], name='reward')
            self.discount_ = tf.placeholder(tf.float32, [None, 1], name='discount')
            self.bootstrap_ = tf.placeholder(tf.float32, [None, 1, action_size], name='bootstrap')

            # concatenate inputs into a single structure
            self.input_ = tf.concat([self.state_, self.joint_action_min_a, self.current_actor, self.joint_action_tm1], 1, name='input')

            # set up COMA critic network (hidden layers)
            self.fc1_critic_ = tf.contrib.layers.fully_connected(self.input_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc2_critic_ = tf.contrib.layers.fully_connected(self.fc1_critic_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc3_critic_ = tf.contrib.layers.fully_connected(self.fc2_critic_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc4_critic_ = tf.contrib.layers.fully_connected(self.fc3_critic_, critic_hidden_size, activation_fn=tf.nn.elu)

            # set up COMA critic network (output layer)
            self.action_values_ = tf.contrib.layers.fully_connected(self.fc4_critic_, action_size, activation_fn=None)

            #reshape output to work with loss function
            self.av_reshape_ = tf.reshape(self.action_values_, [-1, 1, action_size], name='av_reshape')

            # get COMA critic loss
            self.Critic_return = trfl.qlambda(self.av_reshape_, self.actions_, self.reward_, self.discount_, self.bootstrap_, lambda_=lambda_)
            
            # Optimize the loss
            self.critic_loss_ = tf.reduce_mean(self.Critic_return.loss)
            self.critic_optim = tf.train.AdamOptimizer(learning_rate=critic_learning_rate).minimize(self.critic_loss_)

    # used to pass parameters to target network
    def get_network_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]

# individual critics class
class IndCriticNetwork:
    def __init__(self, name, critic_hidden_size=32, critic_learning_rate = 0.0001):
        with tf.variable_scope(name, "Critic"):

            # define inputs for individual critic networks
            self.name=name
            self.input_ = tf.placeholder(tf.float32, [None, obs_size], name='inputs')
            self.reward_ = tf.placeholder(tf.float32, [None, 1], name='reward')
            self.discount_ = tf.placeholder(tf.float32, [None, 1], name='discount')
            self.bootstrap_ = tf.placeholder(tf.float32, [None], name='bootstrap')

            # set up individual critic network (hidden layers)
            self.fc1_critic_ = tf.contrib.layers.fully_connected(self.input_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc2_critic_ = tf.contrib.layers.fully_connected(self.fc1_critic_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc3_critic_ = tf.contrib.layers.fully_connected(self.fc2_critic_, critic_hidden_size, activation_fn=tf.nn.elu)

            # set up individual critic network (output layer)
            self.baseline_ = tf.contrib.layers.fully_connected(self.fc3_critic_, 1, activation_fn=None)
            
            # get individual critic loss
            self.Critic_return, self.advantage = trfl.sequence_advantage_critic_loss(self.baseline_,
                self.reward_, self.discount_, self.bootstrap_, lambda_=lambda_, 
                baseline_cost=baseline_cost)

            # Optimize the loss
            self.critic_loss_ = tf.reduce_mean(self.Critic_return.loss)
            self.critic_optim = tf.train.AdamOptimizer(learning_rate=critic_learning_rate).minimize(self.critic_loss_)

    # used to pass parameters to target network
    def get_network_variables(self): # have to sort out this for centralised architecture
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]

# initialize replay buffer
critic_replay_buffer = []
actor_replay_buffer = []

# initialize replay buffer indicator
replay_time_step = False

# hyperparameters
train_episodes = 5000  
discount = 0.99

replay_prob = 0.15

actor_hidden_size = 64
critic_hidden_size = 64
individual_critic_hidden_size = 32

actor_learning_rate = 0.001
critic_learning_rate = 0.001
individual_critic_learning_rate = 0.001

beta = 0.1 

baseline_cost = 1 # not used

entropy_cost = 0.01
normalise_entropy = True

lambda_ = 1. # not used

# get action and state sizes respectively
action_size = ch.n_actions
obs_size = ch.N_CU

# reset tensorflow graphs
tf.reset_default_graph()

# instantiate COMA critic network

central_critic = CriticNetwork(name="Critic_Net", obs_size=obs_size, action_size=action_size, critic_hidden_size=critic_hidden_size,
                                         critic_learning_rate=critic_learning_rate)

target_critic_net = CriticNetwork(name="arget_Critic_Net", obs_size=obs_size, action_size=action_size, critic_hidden_size=critic_hidden_size,
                                         critic_learning_rate=critic_learning_rate)

target_critic_update_ops = trfl.update_target_variables(target_critic_net.get_network_variables(), 
                                                                  central_critic.get_network_variables(), tau=0.001)

print('Instantiated Critic Network')

# instantiate actor networks and individual critic networks

D2D_actor_nets = []
individual_central_critics = []
individual_target_critic_nets = []
individual_target_critic_update_ops = []

for i in range(0, ch.N_D2D):
    individual_central_critics.append(IndCriticNetwork(name='individual_Critic_Net_{:.0f}'.format(i),critic_hidden_size=individual_critic_hidden_size,
                                         critic_learning_rate=individual_critic_learning_rate))

    individual_target_critic_nets.append(IndCriticNetwork(name='Target_Critic_Net_{:.0f}'.format(i),critic_hidden_size=individual_critic_hidden_size,
                                         critic_learning_rate=individual_critic_learning_rate))

    individual_target_critic_update_ops.append(trfl.update_target_variables(individual_target_critic_nets[i].get_network_variables(), 
                                                                  individual_central_critics[i].get_network_variables(), tau=0.001))

    print('Instantiated Individual Critic Network {:.0f} of {:.0f}'.format(i+1, ch.N_D2D))


    D2D_actor_nets.append(ActorNetwork(name='a_net_{:.0f}'.format(i), obs_size=obs_size, actor_hidden_size=actor_hidden_size,
                                       actor_learning_rate=actor_learning_rate))

    print('Instantiated Actor Network {:.0f} of {:.0f}'.format(i+1, ch.N_D2D))
    print('\n')

# used for progress reports in console
stats_rewards_list = []
stats_every = 10

# initialize the simulator and get channel gains
g_iB, g_j, G_ij, g_jB, G_j_j, d_ij = ch.reset()

# set initial state to list of 0s (one for each Cellular User) 
state = np.zeros(ch.N_CU)
# initialize joint action excluding agent 'a' as all zeros
joint_action_min_a = np.zeros(ch.N_D2D - 1)
# initialize current agent indicator
a = 0
# initialize joint action at t-1 as all zeros
joint_action_tm1 = np.zeros(ch.N_D2D)

# used in plot smoothing procesdure
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

# start tensorflow session
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    total_reward, ep_length = 0, 0
    total_loss_list, action_list, action_prob_list, bootstrap_list, individual_bootstrap_list  = [], [], [], [], []
    critic_total_loss = []
    rewards_list = []
    collision_var = 0
    D2D_collision_probs = []
    collisions = []
    access_ratios = []
    access_rates = []
    avg_throughput = []
    time_avg_throughput = []
    action_sel_record = []

    # begin training
    for ep in range(1, train_episodes):
        
        # determine whether or not to use replay buffer for this time step
        if replay_prob > np.random.random_sample() and len(critic_replay_buffer) > 1:
            print('replaying_timestep')
            replay_time_step = True
        else:
            replay_time_step = False
            critic_replay_buffer.append([])
            actor_replay_buffer.append([])

        # used to determine the collision probability
        ch.collision_indicator = 0
                     
        # generate action probabilities (from policy net forward pass) and use these to sample an action
        action_probs = []
        power_levels = []
        RB_selections = []
        actions = []

        for i in range(0, ch.N_D2D):
            # get action probabilities from actor network for each agent
            action_probs.append(sess.run(D2D_actor_nets[i].action_prob, feed_dict={D2D_actor_nets[i].input_: np.expand_dims(state,axis=0)}))
            action_probs[i] = action_probs[i][0]
            
            # make a random choice of action based on the action probabilities
            actions.append(np.random.choice(np.arange(len(action_probs[i])), p=action_probs[i]))
            power_levels.append(ch.action_space[actions[i]][0])
            RB_selections.append(ch.action_space[actions[i]][1])

        # get cellular user SINRs based on channel gains and action selections
        CU_SINR = ch.CU_SINR_no_collision(g_iB, power_levels, g_jB, RB_selections)

        # get the next state from the simulator
        next_state = ch.state(CU_SINR)

        # get the D2D SINR based on the next state, channel gains and action selections
        D2D_SINR = ch.D2D_SINR_no_collision(power_levels, g_j, G_ij, G_j_j, RB_selections, next_state)

        # get rewards from simulator
        reward, net, individualist_reward, socialist_reward = ch.D2D_reward_no_collision(D2D_SINR, CU_SINR, RB_selections, d_ij)
        
        # divide rewards by channel bandwidth (to simplify learning)
        reward = reward / 10**10
        net = net / 10**10
        individualist_reward = [i / 10**10 for i in individualist_reward]
        socialist_reward = socialist_reward / 10**10
        
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
            individual_bootstrap_values = []
            target_values = []

            for i in range(0, ch.N_D2D):
                # get copy of actions list and pop 'i'th entry to get joint_action_min_a
                actions_copy = actions.copy()
                actions_copy.pop(i)
                
                # get bootsrap values for individual critics
                individual_bootstrap_values.append(sess.run(individual_target_critic_nets[i].baseline_, feed_dict={
                individual_target_critic_nets[i].input_: np.expand_dims(next_state, axis=0)}))

                # get bootstrap values for COMA critic
                target_values.append( sess.run(target_critic_net.action_values_, feed_dict={
                target_critic_net.state_: np.expand_dims(next_state, axis=0),
                target_critic_net.joint_action_min_a: np.reshape(actions_copy, [-1, ch.N_D2D - 1]),
                target_critic_net.current_actor: np.reshape(i, [-1, 1]),
                target_critic_net.joint_action_tm1: np.reshape(joint_action_tm1, [-1, ch.N_D2D])
                }) )

        # critic forward passes
        action_values = []
        for i in range(0, ch.N_D2D):
            # get copy of actions list and remove 'i'th entry to get joint_action_min_a
            actions_copy = actions.copy()
            actions_copy.pop(i)
            
            # get action values from COMA critic
            action_values.append( sess.run(central_critic.action_values_, feed_dict={
            central_critic.state_: np.expand_dims(next_state, axis=0),
            central_critic.joint_action_min_a: np.reshape(actions_copy, [-1, ch.N_D2D - 1]),
            central_critic.current_actor: np.reshape(i, [-1, 1]),
            central_critic.joint_action_tm1: np.reshape(joint_action_tm1, [-1, ch.N_D2D])
            }) )
        # used for post-training plots
        total_losses = []
        
        # get a sample from the critic replay buffer if this time step is being replayed
        if replay_time_step == True:
            replay_sample_ind = range(0, len(critic_replay_buffer) - 1)
            replay_sample_ind = np.random.choice(replay_sample_ind)
            replay_sample = critic_replay_buffer[replay_sample_ind]

        # critic updates
        for i in range(0, ch.N_D2D):
            
            actions_copy = actions.copy()
            actions_copy.pop(i)
            # update each individual critic
            _, total_loss_individual_critic = sess.run([individual_central_critics[i].critic_optim,
                                            individual_central_critics[i].critic_loss_], feed_dict={
                individual_central_critics[i].input_: np.expand_dims(state, axis=0),
                individual_central_critics[i].reward_: np.reshape(individualist_reward[i], (-1, 1)),
                                                                                                         
                individual_central_critics[i].discount_: np.reshape(discount, (-1, 1)),
                individual_central_critics[i].bootstrap_: np.reshape(individual_bootstrap_values[i], (1,)) 
            })
            
            # update COMA critic
            if replay_time_step == True:
                
                # update COMA critic with sample from replay buffer if time step is being replayed
                _, total_loss_critic = sess.run([central_critic.critic_optim,
                                                central_critic.critic_loss_], feed_dict={
                    central_critic.state_: np.expand_dims(replay_sample[i][0], axis=0),
                    central_critic.joint_action_min_a: np.reshape(replay_sample[i][1], [-1, ch.N_D2D - 1]),
                    central_critic.current_actor: np.reshape(replay_sample[i][2], [-1, 1]),
                    central_critic.joint_action_tm1: np.reshape(replay_sample[i][3], [-1, ch.N_D2D]),
                    central_critic.actions_: np.reshape(replay_sample[i][4], (-1, 1)),
                    central_critic.reward_: np.reshape(replay_sample[i][5], (-1, 1)), 
                    central_critic.discount_: np.reshape(replay_sample[i][6], (-1, 1)),
                    central_critic.bootstrap_: np.reshape(replay_sample[i][7], (-1, 1, action_size))
                })
            else:
                # update critic normally
                _, total_loss_critic = sess.run([central_critic.critic_optim,
                                                central_critic.critic_loss_], feed_dict={
                    central_critic.state_: np.expand_dims(state, axis=0),
                    central_critic.joint_action_min_a: np.reshape(actions_copy, [-1, ch.N_D2D - 1]),
                    central_critic.current_actor: np.reshape(i, [-1, 1]),
                    central_critic.joint_action_tm1: np.reshape(joint_action_tm1, [-1, ch.N_D2D]),
                    central_critic.actions_: np.reshape(actions[i], (-1, 1)),
                    central_critic.reward_: np.reshape(socialist_reward, (-1, 1)),
                    central_critic.discount_: np.reshape(discount, (-1, 1)),
                    central_critic.bootstrap_: np.reshape(target_values, (-1, 1, action_size))
                })

                # add the inputs to the COMA critic update for this time step to the critic replay buffer if time step WAS NOT replayed
                critic_replay_buffer[len(critic_replay_buffer)-1].append([state, actions_copy, i, joint_action_tm1, actions[i], socialist_reward, discount, target_values])

        # used for debugging / post training plots
        critic_total_loss.append(total_loss_critic)
        
        # get sample from actor replay buffer if this time step is being replayed
        if replay_time_step == True:
            replay_sample = actor_replay_buffer[replay_sample_ind]
        
        # perform actor network updates
        for i in range(0, ch.N_D2D):
            # initialize 'base' used in counterfactual baseline computation
            base = 0

            # get TD error from individual critics
            seq_i_c_return_, individualist_advantage = sess.run([individual_central_critics[i].Critic_return, individual_central_critics[i].advantage], feed_dict={
                individual_central_critics[i].input_: np.expand_dims(state, axis=0),
                individual_central_critics[i].reward_: np.reshape(individualist_reward[i], (-1, 1)),
                individual_central_critics[i].discount_: np.reshape(discount, (-1, 1)),
                individual_central_critics[i].bootstrap_: np.reshape(individual_bootstrap_values[i], (1,))
            })
            

            if replay_time_step == True:
                # update actor networks with sample from actor replay buffer if time step is being replayed
                _, total_loss, seq_aac_return = sess.run([D2D_actor_nets[i].ac_optim,
                                                          D2D_actor_nets[i].ac_loss, 
                                                          D2D_actor_nets[i].Actor_return], feed_dict={
                    D2D_actor_nets[i].input_: np.expand_dims(replay_sample[i][0], axis=0),
                    D2D_actor_nets[i].action_: np.reshape(replay_sample[i][1], (-1, 1)),
                    D2D_actor_nets[i].action_values_: np.reshape(replay_sample[i][2], (-1, 1))
                })
            else:
                # get sum(Q(s, u) * pi(u|s))
                for j in range(0, action_size):
                    base += action_probs[i][j] * action_values[i][0][j] 

                # remove current actor's 'Q(s, a) * pi(a|s)' from base
                base = base - (action_values[i][0][actions[i]] * action_probs[i][actions[i]])

                # subtract base from action value of current actor's action to get A(s,u)
                adv = action_values[i][0][actions[i]] - base

                # perform beta mix between TD error and A(s,u)
                advantage = ((1 - beta) * individualist_advantage[0][0]) + (beta * adv)

                # update actor networks with 'advantage' as baseline
                _, total_loss, seq_aac_return = sess.run([D2D_actor_nets[i].ac_optim,
                                                                  D2D_actor_nets[i].ac_loss, 
                                                                  D2D_actor_nets[i].Actor_return], feed_dict={
                    D2D_actor_nets[i].input_: np.expand_dims(state, axis=0),
                    D2D_actor_nets[i].action_: np.reshape(actions[i], (-1, 1)),
                    D2D_actor_nets[i].action_values_: np.reshape(advantage, (-1, 1))
                })
                
                # add the inputs to the actor update for this time step to the actor replay buffer if time step WAS NOT replayed
                actor_replay_buffer[len(critic_replay_buffer)-1].append([state, actions[i], advantage])
            
            # used for debugging / post training plots
            total_losses.append(total_loss)
        total_loss_list.append(np.mean(total_losses))

        
        # update COMA critic target network with params from COMA critic
        sess.run(target_critic_update_ops)
        for i in range(0, ch.N_D2D):
            # update individual target networks with params from respective individual critics
            sess.run(individual_target_critic_update_ops[i])
        
        # update joint action at t-1
        joint_action_tm1 = actions 

        # update state
        state = next_state

        # ALL FOLLOWING CODE IS FOR DEBUGGING / POST-TRAINING PLOTS
        a = list(ch.accessed_CUs)
        if 2 in a:
            a = a.index(2)
            b = RB_selections.index(a)
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

        action_sel_record.append(actions)

        if ep % stats_every == 0 or ep == 1:
            print('Power Levels: ', power_levels)
            print('RB Selections: ', RB_selections)
            print('Accessed CUs: ', ch.accessed_CUs)
            print('Rewards of agents: ', reward)
            print('Number of Collisions: ', ch.collision_indicator)
            print('||(Ep)isode: {}|| '.format(ep),
                  'Last net (r)eward: {:.3f}| '.format(net),
                  'Throughput: {:.3f}| '.format(sum(throughput)),
                  'crit (L)oss: {:.4f}|'.format(np.mean(critic_total_loss)),
                  'actor (L)oss: {:.4f}|'.format(np.mean(total_loss_list)))
        stats_rewards_list.append((ep, total_reward, ep_length))
        rewards_list.append(net)

        if len(critic_replay_buffer) > 500:
            critic_replay_buffer.pop(0)

        if len(actor_replay_buffer) > 500:
            actor_replay_buffer.pop(0)
    eps, rews, lens = np.array(stats_rewards_list).T
    smoothed_rews = running_mean(rewards_list, 100)
    smoothed_col_probs = running_mean(D2D_collision_probs, 100)
    smoothed_access_rates = running_mean(access_rates, 100)

    smoothed_throughput = running_mean(time_avg_throughput, 100)

    sr = open('COMApc{}Rew.json'.format(beta*10), 'w+')
    json.dump(list(smoothed_rews), sr)

    sc = open('COMApc{}Col.json'.format(beta*10), 'w+')
    json.dump(list(smoothed_col_probs), sc)

    sa = open('COMApc{}Acc.json'.format(beta*10), 'w+')
    json.dump(list(smoothed_access_rates), sa)
    
    st = open('COMApc{}Thr.json'.format(beta*10), 'w+')
    json.dump(list(smoothed_throughput), st)

    reward_fig = plt.figure()

    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rewards_list, color='grey', alpha=0.3)
    plt.xlabel('Time-slot')
    plt.ylabel('Reward')
    plt.show()

    collision_prob_fig = plt.figure()
    plt.ylim(0, 1)
    plt.xlim(0, train_episodes)
    plt.plot(eps[-len(smoothed_col_probs):], smoothed_col_probs)
    plt.plot(eps, D2D_collision_probs, color='grey', alpha=0.3)
    plt.xlabel('Time-slot')
    plt.ylabel('D2D collision probability')
    plt.show()

    true_collisions_fig = plt.figure()
    plt.ylim(0, ch.N_D2D)
    plt.xlim(0, train_episodes)
    plt.plot(eps, collisions)
    plt.ylabel('Number of collisions')
    plt.xlabel('Time-slot')
    plt.show()

    access_rate_fig = plt.figure()
    plt.ylim(0, 1)
    plt.xlim(0, train_episodes)
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
    
    loss_plot = plt.figure()
    plt.xlim(0, train_episodes)
    plt.plot(eps[-len(critic_total_loss):], critic_total_loss, color='red')
    plt.xlabel('Time-slot')
    plt.ylabel('Critic Loss')
    plt.show()

    pow_sel_plot = plt.figure()
    plt.xlim(0, train_episodes)
    for i in range(0, ch.N_D2D):
        plt.plot(eps[-len(action_sel_record):], [item[i] for item in action_sel_record])
    plt.xlabel('Time-slot')
    plt.ylabel('Action Selection')
    plt.show()
