import tensorflow as tf
import sys
import numpy as np
import tensorflow_probability as tfp
import trfl
import matplotlib.pyplot as plt
import D2D_env_discrete as D2D
import statistics as stats

writer = tf.summary.FileWriter("/home/stefan/tmp/D2D/2")

ch = D2D.Channel()

# set up Actor and Critic networks

class ActorNetwork:
    def __init__(self, name, obs_size=2, actor_hidden_size=32, pow_learning_rate=0.001, RB_learning_rate=0.001, entropy_cost=0.1, normalise_entropy=True):
    
        with tf.variable_scope(name, "Actor"):
            
            self.name=name
            self.input_ = tf.placeholder(tf.float32, [None, obs_size], name='inputs')
            self.action_RB_ = tf.placeholder(tf.int32, [None, 1], name='action_RB')
            self.action_pow_ = tf.placeholder(tf.int32, [None, 1], name='action_pow')
            self.advantage_ = tf.placeholder(tf.float32, [None, 1], name='advantage')

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

            # get loss through power selection distribution logits
            self.Actor_return_pow = trfl.sequence_advantage_actor_loss(self.advantage_, self.policy_logits_power_, self.action_pow_,
                entropy_cost=entropy_cost, normalise_entropy=normalise_entropy)
            
            # get loss through RB selection distribution logits
            self.Actor_return_RB = trfl.sequence_advantage_actor_loss(self.advantage_, self.policy_logits_RB_, self.action_RB_,
                entropy_cost=entropy_cost, normalise_entropy=normalise_entropy)

            # Optimize the loss
            self.ac_loss_pow_ = tf.reduce_mean(self.Actor_return_pow.loss)
            self.ac_loss_RB_ = tf.reduce_mean(self.Actor_return_RB.loss)
            self.ac_optim_pow_ = tf.train.AdamOptimizer(learning_rate=pow_learning_rate).minimize(self.ac_loss_pow_)
            self.ac_optim_RB_ = tf.train.AdamOptimizer(learning_rate=RB_learning_rate).minimize(self.ac_loss_RB_)

    def get_network_variables(self): # have to sort out this for centralised architecture
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


class CriticNetwork:
    def __init__(self, name, critic_hidden_size=32, critic_learning_rate = 0.0001):
        with tf.variable_scope(name, "Critic"):
            
            self.name=name
            self.input_ = tf.placeholder(tf.float32, [None, obs_size], name='inputs')
            self.reward_ = tf.placeholder(tf.float32, [None, 1], name='reward')
            self.discount_ = tf.placeholder(tf.float32, [None, 1], name='discount')
            self.bootstrap_ = tf.placeholder(tf.float32, [None], name='bootstrap')

            # set up critic network
            self.fc1_critic_ = tf.contrib.layers.fully_connected(self.input_, critic_hidden_size, activation_fn=tf.nn.elu)

            self.fc2_critic_ = tf.contrib.layers.fully_connected(self.fc1_critic_, critic_hidden_size, activation_fn=tf.nn.elu)

            self.baseline_ = tf.contrib.layers.fully_connected(self.fc2_critic_, 1, activation_fn=None)

            self.Critic_return, self.advantage = trfl.sequence_advantage_critic_loss(self.baseline_,
                self.reward_, self.discount_, self.bootstrap_, lambda_=lambda_, 
                baseline_cost=baseline_cost)
            
            self.critic_loss_ = tf.reduce_mean(self.Critic_return.loss)
            self.critic_optim = tf.train.AdamOptimizer(learning_rate=critic_learning_rate).minimize(self.critic_loss_)
    
    def get_advantage(self):
        return self.advantage

    def get_network_variables(self): # have to sort out this for centralised architecture
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


# hyperparameters
train_episodes = 5000  
discount = 0.99

actor_hidden_size = 64
critic_hidden_size = 32
    
pow_learning_rate = 0.001
RB_learning_rate = 0.001

critic_learning_rate = 0.0001

target_pow_learning_rate = 0.001
target_RB_learning_rate = 0.001

baseline_cost = 1 #scale derivatives between actor and critic networks

# entropy hyperparameters
entropy_cost = 0.01
normalise_entropy = True

# one step returns ie TD(0).
lambda_ = 1.

action_size = ch.n_actions
obs_size = ch.N_CU

print('action_size: ', action_size)
print('obs_size: ', obs_size)

tf.reset_default_graph()

# instantiate critic network

central_critic = CriticNetwork(name="Critic_Net",critic_hidden_size=critic_hidden_size,
                                         critic_learning_rate=critic_learning_rate)

target_critic_net = CriticNetwork(name="Target_Critic_Net",critic_hidden_size=critic_hidden_size,
                                         critic_learning_rate=critic_learning_rate)

target_critic_update_ops = trfl.update_target_variables(target_critic_net.get_network_variables(), 
                                                                  central_critic.get_network_variables(), tau=0.001)

print('Instantiated Critic Network')

# instantialte actor nets

D2D_actor_nets = []

for i in range(0, ch.N_D2D):
    D2D_actor_nets.append(ActorNetwork(name='a_net_{:.0f}'.format(i), obs_size=obs_size, actor_hidden_size=actor_hidden_size,
                                       pow_learning_rate=pow_learning_rate, RB_learning_rate=RB_learning_rate,
                                       entropy_cost=entropy_cost, normalise_entropy=normalise_entropy))

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
    total_loss_list_pow, total_loss_list_RB, action_list, action_prob_list, bootstrap_list = [], [], [], [], []
    rewards_list = []
    collision_var = 0
    D2D_collision_probs = []
    collisions = []
    access_ratios = []
    access_rates = []
    avg_throughput = []
    time_avg_throughput = []

    for ep in range(1, train_episodes):

        ch.collision_indicator = 0
                     
        # generate action probabilities from policy net and sample from the action probs
        power_action_probs = []
        RB_action_probs = []
        pow_sels = []
        RB_sels = []

        unused_action_probs_pow = []
        unused_action_probs_rb = []

        unused_actions_pow = []
        unused_actions_rb = []

        for i in range(0, ch.N_D2D):
            power_action_probs.append(sess.run(D2D_actor_nets[i].action_prob_power_, feed_dict={D2D_actor_nets[i].input_: np.expand_dims(state,axis=0)}))
            power_action_probs[i] = power_action_probs[i][0]

            RB_action_probs.append(sess.run(D2D_actor_nets[i].action_prob_RB_, feed_dict={D2D_actor_nets[i].input_: np.expand_dims(state,axis=0)}))
            RB_action_probs[i] = RB_action_probs[i][0]
            
            pow_copy = list(ch.power_levels).copy()
            rb_copy = list(ch.CU_index).copy()

            pow_sels.append(np.random.choice(ch.power_levels, p=power_action_probs[i]))
            RB_sels.append(np.random.choice(ch.CU_index, p=RB_action_probs[i]))

            power_action_probs[i] = list(power_action_probs[i])
            RB_action_probs[i] = list(RB_action_probs[i])

            unused_action_probs_pow.append(power_action_probs[i].copy())
            unused_action_probs_rb.append(RB_action_probs[i].copy())

            # remove action choices from unused action prob lists
            unused_action_probs_pow[i].pop(pow_copy.index(pow_sels[i]))
            unused_action_probs_rb[i].pop(rb_copy.index(RB_sels[i]))

            # remove action choices from unused action proper lists
            pow_copy.remove(pow_sels[i])
            rb_copy.remove(RB_sels[i])

            unused_actions_pow.append(pow_copy)
            unused_actions_rb.append(rb_copy)

        #print("power_levels: ", ch.power_levels)
        CU_SINR = ch.CU_SINR_no_collision(g_iB, pow_sels, g_jB, RB_sels)

        # Q VALUE CALCULATIONS (unused actions) -------------------------------------------------------------------------------------

        # get s' and r' for all possible other actions, for each agent
        pow_alternate_states = []
        pow_alt_rewards = []

        rb_alternate_states = []
        rb_alt_rewards = []

        for i in range(0, ch.N_D2D):
            pow_alternate_states.append([])
            rb_alternate_states.append([])

            pow_alt_rewards.append([])
            rb_alt_rewards.append([])

            temp_pow = pow_sels.copy() # a = [1,2,3]
            temp_rb = RB_sels.copy()   # b = [1,2,3]

            temp_pow.pop(i)            # a = [2,3]
            temp_rb.pop(i)             # b = [2,3]
            for j in range(0, len(unused_actions_pow[i])):   #      i
                temp_pow.insert(i, unused_actions_pow[i][j]) # a = [j,2,3]
                pow_alternate_states[i].append(ch.state(ch.CU_SINR_no_collision(g_iB, temp_pow, g_jB, RB_sels)))
                D2D_SINR = ch.D2D_SINR_no_collision(temp_pow, g_j, G_ij, G_j_j, RB_sels, pow_alternate_states[i])
                pow_alt_rewards[i].append(ch.D2D_reward_no_collision(D2D_SINR, CU_SINR, RB_sels, d_ij))
                temp_pow.pop(i)
            temp_pow.insert(i, pow_sels[i])

            for j in range(0, len(unused_actions_rb[i])):
                temp_rb.insert(i, unused_actions_rb[i][j])
                rb_alternate_states[i].append(ch.state(ch.CU_SINR_no_collision(g_iB, pow_sels, g_jB, temp_rb)))
                D2D_SINR = ch.D2D_SINR_no_collision(pow_sels, g_j, G_ij, G_j_j, temp_rb, rb_alternate_states[i])
                rb_alt_rewards[i].append(ch.D2D_reward_no_collision(D2D_SINR, CU_SINR, RB_sels, d_ij))
                temp_rb.pop(i)
            temp_rb.insert(i, RB_sels[i])

            # CHECK THAT THESE WORK

        # feed states through critic to get V(s') CHANGE TO WORK

        alternate_values_pow = []
        alternate_values_rb = []

        for i in range(0, ch.N_D2D):
            alternate_values_pow.append([])
            alternate_values_rb.append([])

            for j in range(0, len(pow_alternate_states[i])):
                alternate_values_pow[i].append(sess.run(central_critic.baseline_, feed_dict={
                    central_critic.input_: np.reshape(pow_alternate_states[i][j], [-1, ch.N_CU])
                }))
        
            for j in range(0, len(rb_alternate_states[i])):
                alternate_values_rb[i].append(sess.run(central_critic.baseline_, feed_dict={
                    central_critic.input_: np.reshape(rb_alternate_states[i][j], [-1, ch.N_CU])
                }))

        # compute q values for all alternate actions

        alternate_qs_pow = []
        alternate_qs_rb = []

        for i in range(0, ch.N_D2D):
            alternate_qs_pow.append([])
            for j in range(0, len(alternate_values_pow[i])):
                alternate_qs_pow[i].append((pow_alt_rewards[i][j][1] / 10**10) + (discount * alternate_values_pow[i][j][0][0]))
        
        for i in range(0, ch.N_D2D):
            alternate_qs_rb.append([])
            for j in range(0, len(alternate_values_rb[i])):
                alternate_qs_rb[i].append((rb_alt_rewards[i][j][1] / 10**10) + (discount * alternate_values_rb[i][j][0][0]))
        
        # multiply alternate q values by respective action probs given by policy output distributution

        alternate_weighted_qs_pow = []
        alternate_weighted_qs_rb = []

        for i in range(0, ch.N_D2D):
            alternate_weighted_qs_pow.append([])
            alternate_weighted_qs_rb.append([])
            for j in range(0, len(alternate_qs_pow[i])):
                alternate_weighted_qs_pow[i].append(unused_action_probs_pow[i][j] * alternate_qs_pow[i][j])

            for j in range(0, len(alternate_qs_rb[i])):
                alternate_weighted_qs_rb[i].append(unused_action_probs_rb[i][j] * alternate_qs_rb[i][j])
        
        # sum calculations for each agent

        alt_pow_sums = []
        alt_rb_sums = []

        for i in range(0, ch.N_D2D):
            alt_pow_sums.append(sum(alternate_weighted_qs_pow[i]))
            alt_rb_sums.append(sum(alternate_weighted_qs_rb[i]))

        #-----------------------------------------------------------------------------------------------------------

        next_state_true = ch.state(CU_SINR)
        next_state = ch.accessed_CUs

        D2D_SINR = ch.D2D_SINR_no_collision(pow_sels, g_j, G_ij, G_j_j, RB_sels, next_state_true)
        reward, net, ind_rew, soc_rew = ch.D2D_reward_no_collision(D2D_SINR, CU_SINR, RB_sels, d_ij)
            
        reward = reward / 10**10
        net = net / 10**10
        
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
          bootstrap_values = []
          for i in range(0, ch.N_D2D):
            bootstrap_values.append(sess.run(target_critic_net.baseline_, feed_dict={
                target_critic_net.input_: np.expand_dims(next_state, axis=0)}))
        
        #train networks
        
        total_losses_pow = []
        seq_aac_returns_pow = []

        total_losses_RB = []
        seq_aac_returns_RB = []
        
        # critic update

        ind_q = []
        soc_q = []

        for i in range(0, ch.N_D2D):
            ind_q.append(net + (discount * bootstrap_values[i]))
            soc_q.append(net + (discount * bootstrap_values[i]))

        _, total_loss_critic = sess.run([central_critic.critic_optim,
                                        central_critic.critic_loss_], feed_dict={
            central_critic.input_: np.expand_dims(state, axis=0),
            central_critic.reward_: np.reshape(stats.mean(reward), (-1, 1)), # taking the mean reward is the naive solution as
                                                                             # it fails to address the credit assignment problem
            central_critic.discount_: np.reshape(discount, (-1, 1)),
            central_critic.bootstrap_: np.reshape(bootstrap_values[i], (1,)) 
        })
         # need to change the advantage so as to address this problem - produce an advantage for each agent that
         # gives insight into that specific agent's individual contribution to the global reward (global reward being the cumulative SINR of CUs)
        

        for i in range(0, ch.N_D2D):
            
            # get advantage for jth D2D

            seq_c_return_, advantage = sess.run([central_critic.Critic_return, central_critic.advantage], feed_dict={
                central_critic.input_: np.expand_dims(state, axis=0),
                central_critic.reward_: np.reshape(reward[i], (-1, 1)),
                central_critic.discount_: np.reshape(discount, (-1, 1)),
                central_critic.bootstrap_: np.reshape(bootstrap_values[i], (1,)) #np.expand_dims(bootstrap_value, axis=0)
            })
            
            # power level selection distribution based updates

            _, total_loss_pow, seq_aac_return_pow = sess.run([D2D_actor_nets[i].ac_optim_pow_,
                                                              D2D_actor_nets[i].ac_loss_pow_, 
                                                              D2D_actor_nets[i].Actor_return_pow], feed_dict={
                D2D_actor_nets[i].input_: np.expand_dims(state, axis=0),
                D2D_actor_nets[i].action_pow_: np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
                D2D_actor_nets[i].action_RB_: np.reshape(RB_sels[i], (-1, 1)),
                D2D_actor_nets[i].advantage_: np.reshape((ind_q[i] - alt_pow_sums[i]), (-1, 1)),
            })

            total_losses_pow.append(total_loss_pow)
            seq_aac_returns_pow.append(seq_aac_return_pow)

            _, total_loss_RB, seq_aac_return_RB = sess.run([D2D_actor_nets[i].ac_optim_RB_,
                                                            D2D_actor_nets[i].ac_loss_RB_,
                                                            D2D_actor_nets[i].Actor_return_RB], feed_dict={
                D2D_actor_nets[i].input_: np.expand_dims(state, axis=0),
                D2D_actor_nets[i].action_pow_: np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
                D2D_actor_nets[i].action_RB_: np.reshape(RB_sels[i], (-1, 1)),
                D2D_actor_nets[i].advantage_: np.reshape((soc_q[i] - alt_rb_sums[i]), (-1, 1)),
            })

            total_losses_RB.append(total_loss_RB)
            seq_aac_returns_RB.append(seq_aac_return_RB)

        total_loss_list_pow.append(np.mean(total_losses_pow))
        total_loss_list_RB.append(np.mean(total_losses_RB))
        
        #update target network
        for i in range(0, ch.N_D2D):
            sess.run(target_critic_update_ops)
        
        #action_list.append(actions)
        bootstrap_list.append(bootstrap_values)
        #action_prob_list.append(action_probs)
        
        #if total_reward < -250:
        #  done = 1

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
            #for i in range(0, ch.N_D2D):
            #print('Last State: ', state)
            print('Power Levels: ', pow_sels)
            print('RB Selections: ', RB_sels)
            print('Accessed CUs: ', ch.accessed_CUs)
            print('Rewards of agents: ', reward)
            print('Number of Collisions: ', ch.collision_indicator)
            print('||(Ep)isode: {}|| '.format(ep),
                  #'(T)otal reward: {:.5f}| '.format(np.mean(stats_rewards_list[-stats_every:],axis=0)[1]),
                  'Last net (r)eward: {:.3f}| '.format(net),
                  #'Ep length: {:.1f}| '.format(np.mean(stats_rewards_list[-stats_every:],axis=0)[2]),
                  'Throughput: {:.3f}| '.format(sum(throughput)),
                  'Pow (L)oss: {:4f}|'.format(np.mean(total_loss_list_pow)),
                  'RB (L)oss: {:4f}|'.format(np.mean(total_loss_list_RB)))
                  #'Actor loss: {:.5f}'.format(stats_actor_loss),
                  #'Critic loss: {:.5f}'.format(stats_critic_loss))
            #print(np.mean(bootstrap_value), np.mean(action_list), np.mean(action_prob_list,axis=0))
        #stats_actor_loss, stats_critic_loss = 0, 0
        #total_loss_list = []
        stats_rewards_list.append((ep, total_reward, ep_length))
        rewards_list.append(net)
        state = next_state

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
    

