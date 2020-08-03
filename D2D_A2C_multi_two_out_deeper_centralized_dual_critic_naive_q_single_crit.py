import tensorflow as tf
import sys
import numpy as np
import tensorflow_probability as tfp
import trfl
import matplotlib.pyplot as plt
import D2D_env_discrete as D2D
import statistics as stats
import collections

writer = tf.summary.FileWriter("/home/stefan/tmp/D2D/2")

ch = D2D.Channel()

# set up Actor and Critic networks

class ActorNetwork:
    def __init__(self, name, obs_size=2, actor_hidden_size=32, actor_learning_rate=0.001):
    
        with tf.variable_scope(name, "Actor"):
            
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

            # get loss through power selection distribution logits
            self.Actor_return = trfl.discrete_policy_gradient(self.policy_logits, self.action_, self.action_values_)
            
            # get loss through RB selection distribution logits

            # Optimize the loss
            self.ac_loss = tf.reduce_mean(self.Actor_return)
            self.ac_optim = tf.train.AdamOptimizer(learning_rate=actor_learning_rate).minimize(self.ac_loss)


    def get_network_variables(self): # have to sort out this for centralised architecture
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


class CriticNetwork:
    def __init__(self, name, obs_size=None, action_size=None, critic_hidden_size=32, critic_learning_rate = 0.0001):
        with tf.variable_scope(name, "Critic"):
            
            self.name=name
            self.state_ = tf.placeholder(tf.float32, [None, obs_size], name='state')
            self.joint_action_min_a = tf.placeholder(tf.float32, [None, ch.N_D2D - 1], name='joint_actions_min_a')
            self.current_actor = tf.placeholder(tf.float32, [None, 1], name='current_actor')
            self.joint_action_tm1 = tf.placeholder(tf.float32, [None, ch.N_D2D], name='joint_action_tm1')

            self.actions_ = tf.placeholder(tf.int32, [None, 1], name='c_actions')
            self.reward_ = tf.placeholder(tf.float32, [None, 1], name='reward')
            self.discount_ = tf.placeholder(tf.float32, [None, 1], name='discount')
            self.bootstrap_ = tf.placeholder(tf.float32, [None, 1, action_size], name='bootstrap')

            self.input_ = tf.concat([self.state_, self.joint_action_min_a, self.current_actor, self.joint_action_tm1], 1, name='input')

            # set up critic network
            self.fc1_critic_ = tf.contrib.layers.fully_connected(self.input_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc2_critic_ = tf.contrib.layers.fully_connected(self.fc1_critic_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc3_critic_ = tf.contrib.layers.fully_connected(self.fc2_critic_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc4_critic_ = tf.contrib.layers.fully_connected(self.fc3_critic_, critic_hidden_size, activation_fn=tf.nn.elu)

            self.action_values_ = tf.contrib.layers.fully_connected(self.fc4_critic_, action_size, activation_fn=None)

            self.av_reshape_ = tf.reshape(self.action_values_, [-1, 1, action_size], name='av_reshape')

            self.Critic_return = trfl.qlambda(self.av_reshape_, self.actions_, self.reward_, self.discount_, self.bootstrap_, lambda_=lambda_)
            
            self.critic_loss_ = tf.reduce_mean(self.Critic_return.loss)
            self.critic_optim = tf.train.AdamOptimizer(learning_rate=critic_learning_rate).minimize(self.critic_loss_)

    def get_network_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


# hyperparameters
train_episodes = 5000  
discount = 0.99

critic_replay_buffer = []
actor_replay_buffer = []
replay_prob = 0.1
replay_time_step = False

actor_hidden_size = 64
critic_hidden_size = 64

actor_learning_rate = 0.001

critic_learning_rate = 0.001

target_pow_learning_rate = 0.001
target_RB_learning_rate = 0.001

beta = 0.1 # socialism parameter (Lower - More Individualist | Higher - More Socialist)

baseline_cost = 1 #scale derivatives between actor and critic networks

# entropy hyperparameters
entropy_cost = 0.01
normalise_entropy = True

# one step returns ie TD(0).
lambda_ = 1.

action_size = ch.n_actions

#action_size_pow = ch.D2D_tr_Power_levels
#action_size_RB = ch.N_CU
obs_size = ch.N_CU

#print('action_size_pow: ', action_size_pow)
#print('action_size_RB: ', action_size_RB)

print('action_size: ', action_size)
print('obs_size: ', obs_size)

tf.reset_default_graph()

# instantiate social critic networks

central_critic = CriticNetwork(name="Critic_Net", obs_size=obs_size, action_size=action_size, critic_hidden_size=critic_hidden_size,
                                         critic_learning_rate=critic_learning_rate)

target_critic_net = CriticNetwork(name="arget_Critic_Net", obs_size=obs_size, action_size=action_size, critic_hidden_size=critic_hidden_size,
                                         critic_learning_rate=critic_learning_rate)

target_critic_update_ops = trfl.update_target_variables(target_critic_net.get_network_variables(), 
                                                                  central_critic.get_network_variables(), tau=0.001)

print('Instantiated Critic Network')

# instantiate actor nets

D2D_actor_nets = []

for i in range(0, ch.N_D2D):
    D2D_actor_nets.append(ActorNetwork(name='a_net_{:.0f}'.format(i), obs_size=obs_size, actor_hidden_size=actor_hidden_size,
                                       actor_learning_rate=actor_learning_rate))

    print('Instantiated Actor Network {:.0f} of {:.0f}'.format(i+1, ch.N_D2D))
    print('\n')


stats_rewards_list = []
stats_every = 10

initial_actions = []
#power_levels = []
#RB_selections = []

g_iB, g_j, G_ij, g_jB, G_j_j, d_ij = ch.reset()
#for i in range(0, ch.N_D2D):
#    action = np.random.randint(0, 299, 1)
#    power_levels.append(ch.action_space[action][0][0])
#    RB_selections.append(ch.action_space[action][0][1])
    
#print(power_levels)
#print(RB_selections)

state = np.zeros(ch.N_CU)
joint_action_min_a = np.zeros(ch.N_D2D - 1)
a = 0
joint_action_tm1 = np.zeros(ch.N_D2D)

#CU_SINR = ch.CU_SINR_no_collision(g_iB, power_levels, g_jB, RB_selections)

#print(CU_SINR)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    total_reward, ep_length, done = 0, 0, 0
    stats_actor_loss_pow, stats_critic_loss_pow = 0., 0.
    stats_actor_loss_RB, stats_critic_loss_RB = 0., 0.
    total_loss_list, total_loss_list_RB, action_list, action_prob_list, bootstrap_list, individual_bootstrap_list  = [], [], [], [], [], []
    critic_total_loss = []
    rewards_list = []
    collision_var = 0
    D2D_collision_probs = []
    collisions = []
    access_ratios = []
    access_rates = []
    avg_throughput = []
    time_avg_throughput = []
    pow_sel_record = []
    RB_sel_record = []

    for ep in range(1, train_episodes):
        
        replay_time_step = False
        if replay_prob > np.random.random_sample() and len(critic_replay_buffer) > 1:
            print('replaying_timestep')
            replay_time_step = True

        critic_replay_buffer.append([])
        actor_replay_buffer.append([])

        ch.collision_indicator = 0
                     
        # generate action probabilities (from policy net forward pass) and use these to sample an action
        action_probs = []
        power_levels = []
        RB_selections = []
        actions = []
        RB_sels_ind = []
        unused_actions = []
        ua_probs = []

        for i in range(0, ch.N_D2D):
            action_probs.append(sess.run(D2D_actor_nets[i].action_prob, feed_dict={D2D_actor_nets[i].input_: np.expand_dims(state,axis=0)}))
            action_probs[i] = action_probs[i][0]
            
            actions.append(np.random.choice(np.arange(len(action_probs[i])), p=action_probs[i]))
            power_levels.append(ch.action_space[actions[i]][0])
            RB_selections.append(ch.action_space[actions[i]][1])
            #action_sels.append(np.random.choice(ch.power_levels, p=action_probs[i]))
            #RB_sels.append(np.random.choice(ch.CU_index, p=RB_action_probs[i]))

            #pow_sels_ind.append(list(ch.power_levels).index(pow_sels[i]))
            #RB_sels_ind.append(list(ch.CU_index).index(RB_sels[i]))

        #print('pow_sels', pow_sels_ind)
        #print("power_levels: ", ch.power_levels)
        CU_SINR = ch.CU_SINR_no_collision(g_iB, power_levels, g_jB, RB_selections)

        next_state = ch.state(CU_SINR)
        #next_state = ch.accessed_CUs

        D2D_SINR = ch.D2D_SINR_no_collision(power_levels, g_j, G_ij, G_j_j, RB_selections, next_state)
        reward, net, individualist_reward, socialist_reward = ch.D2D_reward_no_collision(D2D_SINR, CU_SINR, RB_selections, d_ij)
            
        reward = reward / 10**10
        net = net / 10**10
        individualist_reward = [i / 10**10 for i in individualist_reward]
        socialist_reward = socialist_reward / 10**10
        
        if ch.collision_indicator > 0:
            collision_var += 1
        
        collisions.append(ch.collision_indicator)
        
        D2D_collision_probs.append(collision_var / ep)

        next_state = np.clip(next_state,-1.,1.)
        total_reward += net

        ep_length += 1

        if ep == train_episodes:
            bootstrap_value = np.zeros((1,),dtype=np.float32)
        else:
            #get bootstrap value
          
            individual_bootstrap_values = []

            target_values = []

            for i in range(0, ch.N_D2D):
                actions_copy = actions.copy()

                actions_copy.pop(i)

                target_values.append( sess.run(target_critic_net.action_values_, feed_dict={
                target_critic_net.state_: np.expand_dims(next_state, axis=0),
                target_critic_net.joint_action_min_a: np.reshape(actions_copy, [-1, ch.N_D2D - 1]),# @@@@@@@@@@ Sort out action selections
                target_critic_net.current_actor: np.reshape(i, [-1, 1]),
                target_critic_net.joint_action_tm1: np.reshape(joint_action_tm1, [-1, ch.N_D2D])
                }) )

        #print(individual_target_values)
        #print(social_target_values)

        # critic forward passes

        action_values = []
        for i in range(0, ch.N_D2D):

            actions_copy = actions.copy()

            actions_copy.pop(i)

            action_values.append( sess.run(central_critic.action_values_, feed_dict={
            central_critic.state_: np.expand_dims(next_state, axis=0),
            central_critic.joint_action_min_a: np.reshape(actions_copy, [-1, ch.N_D2D - 1]), # @@@@@@@@@@ Sort out action selections
            central_critic.current_actor: np.reshape(i, [-1, 1]),
            central_critic.joint_action_tm1: np.reshape(joint_action_tm1, [-1, ch.N_D2D])
            }) )


        #print(individualist_action_values)
        #print(socialist_action_values)

        seq_aac_returns_pow = []
        total_losses = []
        
        if replay_time_step == True:
            replay_sample_ind = range(0, len(critic_replay_buffer) - 1)
            replay_sample_ind = np.random.choice(replay_sample_ind)
            replay_sample = critic_replay_buffer[replay_sample_ind]

        # critic updates
        for i in range(0, ch.N_D2D):
            
            actions_copy = actions.copy()

            actions_copy.pop(i)
            
            if replay_time_step == True:
                #print(replay_sample)

                _, total_loss_critic = sess.run([central_critic.critic_optim,
                                                central_critic.critic_loss_], feed_dict={
                    central_critic.state_: np.expand_dims(replay_sample[i][0], axis=0),
                    central_critic.joint_action_min_a: np.reshape(replay_sample[i][1], [-1, ch.N_D2D - 1]),
                    central_critic.current_actor: np.reshape(replay_sample[i][2], [-1, 1]),
                    central_critic.joint_action_tm1: np.reshape(replay_sample[i][3], [-1, ch.N_D2D]),
                    #central_critic.action_values_: np.reshape(individualist_action_values[i], (-1, action_size_pow)),
                    central_critic.actions_: np.reshape(replay_sample[i][4], (-1, 1)),
                    central_critic.reward_: np.reshape(replay_sample[i][5], (-1, 1)), # taking the mean reward is the naive solution as
                                                                                                            # it fails to address the credit assignment problem
                    central_critic.discount_: np.reshape(replay_sample[i][6], (-1, 1)),
                    central_critic.bootstrap_: np.reshape(replay_sample[i][7], (-1, 1, action_size))
                })
            else:
                _, total_loss_critic = sess.run([central_critic.critic_optim,
                                                central_critic.critic_loss_], feed_dict={
                    central_critic.state_: np.expand_dims(state, axis=0),
                    central_critic.joint_action_min_a: np.reshape(actions_copy, [-1, ch.N_D2D - 1]),
                    central_critic.current_actor: np.reshape(i, [-1, 1]),
                    central_critic.joint_action_tm1: np.reshape(joint_action_tm1, [-1, ch.N_D2D]),
                    #central_critic.action_values_: np.reshape(individualist_action_values[i], (-1, action_size_pow)),
                    central_critic.actions_: np.reshape(actions[i], (-1, 1)),
                    central_critic.reward_: np.reshape(reward[i], (-1, 1)), # taking the mean reward is the naive solution as
                                                                                                            # it fails to address the credit assignment problem
                    central_critic.discount_: np.reshape(discount, (-1, 1)),
                    central_critic.bootstrap_: np.reshape(target_values, (-1, 1, action_size))
                })

            
            critic_replay_buffer[len(critic_replay_buffer)-1].append([state, actions_copy, i, joint_action_tm1, actions[i], reward[i], discount, target_values])
         # need to change the advantage so as to address this problem - produce an advantage for each agent that
         # gives insight into that specific agent's individual contribution to the global reward (global reward being the cumulative SINR of CUs)

        critic_total_loss.append(total_loss_critic)
          
        if replay_time_step == True:
            replay_sample = actor_replay_buffer[replay_sample_ind]
        
        for i in range(0, ch.N_D2D):
            # power level selection distribution based updates
            #print(individualist_action_values[i][0][pow_sels_ind[i]])
            base = 0

            if replay_time_step == True:

                _, total_loss, seq_aac_return = sess.run([D2D_actor_nets[i].ac_optim,
                                                          D2D_actor_nets[i].ac_loss, 
                                                          D2D_actor_nets[i].Actor_return], feed_dict={
                    D2D_actor_nets[i].input_: np.expand_dims(replay_sample[i][0], axis=0),
                    D2D_actor_nets[i].action_: np.reshape(replay_sample[i][1], (-1, 1)),
                    D2D_actor_nets[i].action_values_: np.reshape(replay_sample[i][2], (-1, 1))
                })
            else:

                for j in range(0, len(ch.power_levels)):
                    base += action_probs[i][j] * action_values[i][0][j] 
            

                adv = (2 * action_values[i][0][actions[i]]) - base


                _, total_loss, seq_aac_return = sess.run([D2D_actor_nets[i].ac_optim,
                                                                  D2D_actor_nets[i].ac_loss, 
                                                                  D2D_actor_nets[i].Actor_return], feed_dict={
                    D2D_actor_nets[i].input_: np.expand_dims(state, axis=0),
                    D2D_actor_nets[i].action_: np.reshape(actions[i], (-1, 1)),
                    D2D_actor_nets[i].action_values_: np.reshape(adv, (-1, 1))
                })
            
            actor_replay_buffer[len(critic_replay_buffer)-1].append([state, actions[i], adv])
            
            total_losses.append(total_loss)

        total_loss_list.append(np.mean(total_losses))

        
        #update target network
        sess.run(target_critic_update_ops)
        
        #action_list.append(actions)
        #bootstrap_list.append(social_bootstrap_values)
        #individual_bootstrap_list.append(individual_bootstrap_values)
        #action_prob_list.append(action_probs)
        
        joint_action_tm1 = actions # These might not be rightt (they are the action selections proper, not the action indexes)

        #if total_reward < -250:
        #  done = 1

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

        pow_sel_record.append(actions)

        if ep % stats_every == 0 or ep == 1:
            #for i in range(0, ch.N_D2D):
            #print('Last State: ', state)
            print('Power Levels: ', power_levels)
            print('RB Selections: ', RB_selections)
            print('Accessed CUs: ', ch.accessed_CUs)
            print('Rewards of agents: ', reward)
            print('Number of Collisions: ', ch.collision_indicator)
            print('||(Ep)isode: {}|| '.format(ep),
                  #'(T)otal reward: {:.5f}| '.format(np.mean(stats_rewards_list[-stats_every:],axis=0)[1]),
                  'Last net (r)eward: {:.3f}| '.format(net),
                  #'Ep length: {:.1f}| '.format(np.mean(stats_rewards_list[-stats_every:],axis=0)[2]),
                  'Throughput: {:.3f}| '.format(sum(throughput)),
                  'crit (L)oss: {:.4f}|'.format(np.mean(critic_total_loss)),
                  'actor (L)oss: {:.4f}|'.format(np.mean(total_loss_list)))
                  #'Actor loss: {:.5f}'.format(stats_actor_loss),
                  #'Critic loss: {:.5f}'.format(stats_critic_loss))
            #print(np.mean(bootstrap_value), np.mean(action_list), np.mean(action_prob_list,axis=0))
        #stats_actor_loss, stats_critic_loss = 0, 0
        #total_loss_list = []
        stats_rewards_list.append((ep, total_reward, ep_length))
        rewards_list.append(net)
        state = next_state

        if len(critic_replay_buffer) > 500:
            critic_replay_buffer.pop(0)

        if len(actor_replay_buffer) > 500:
            actor_replay_buffer.pop(0)

        #writer.add_graph(sess.graph)
        #print("Graph added!")
    eps, rews, lens = np.array(stats_rewards_list).T
    smoothed_rews = running_mean(rewards_list, 100)
    smoothed_col_probs = running_mean(D2D_collision_probs, 100)
    smoothed_access_rates = running_mean(access_rates, 100)

#    for i in range(0, len(time_avg_throughput)):
#        time_avg_throughput[i] = time_avg_throughput[i] / train_episodes

    smoothed_throughput = running_mean(time_avg_throughput, 100)

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
        plt.plot(eps[-len(pow_sel_record):], [item[i] for item in pow_sel_record])
    plt.xlabel('Time-slot')
    plt.ylabel('Power Level Selection')
    plt.show()

