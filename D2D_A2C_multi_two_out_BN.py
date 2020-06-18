import tensorflow as tf
import sys
import numpy as np
import tensorflow_probability as tfp
import trfl
import matplotlib.pyplot as plt
import D2D_env_discrete as D2D

writer = tf.summary.FileWriter("/home/stefan/tmp/D2D/2")

ch = D2D.Channel()

# set up Actor and Critic networks
class ActorCriticNetwork:
    def __init__(self, name, obs_size=2, action_size=2, actor_hidden_size=32, critic_hidden_size=32, ac_learning_rate=0.001,  
                   entropy_cost=0.01, normalise_entropy=True, lambda_=0., baseline_cost=1.):
    
        with tf.variable_scope(name):
            # hyperparameter bootstrap_n determines the batch size
            self.name=name
            self.input_ = tf.placeholder(tf.float32, [None, obs_size], name='inputs')
            self.action_RB_ = tf.placeholder(tf.int32, [None, 1], name='action_RB')
            self.action_pow_ = tf.placeholder(tf.int32, [None, 1], name='action_pow')
            self.reward_ = tf.placeholder(tf.float32, [None, 1], name='reward')
            self.discount_ = tf.placeholder(tf.float32, [None, 1], name='discount')
            self.bootstrap_ = tf.placeholder(tf.float32, [None], name='bootstrap')

            # set up actor network
            self.fc1_actor_ = tf.contrib.layers.fully_connected(self.input_, actor_hidden_size, activation_fn=tf.nn.elu)

            self.mean1_actor_, self.variance1_actor_ = tf.nn.moments(self.fc1_actor_, axes=-1)
            self.bn1_actor_ = tf.nn.batch_normalization(self.fc1_actor_, mean=self.mean1_actor_, variance=self.variance1_actor_, offset=0, scale=1, variance_epsilon=0.0001)

            self.fc2_actor_ = tf.contrib.layers.fully_connected(self.bn1_actor_, actor_hidden_size, activation_fn=tf.nn.elu)

            self.mean2_actor_, self.variance2_actor_ = tf.nn.moments(self.fc2_actor_, axes=-1)
            self.bn2_actor_ = tf.nn.batch_normalization(self.fc2_actor_, mean=self.mean2_actor_, variance=self.variance2_actor_, offset=0, scale=1, variance_epsilon=0.0001)

            self.fc3_actor_power_ = tf.contrib.layers.fully_connected(self.bn2_actor_, ch.D2D_tr_Power_levels, activation_fn=None)            
            self.fc3_actor_RB_ = tf.contrib.layers.fully_connected(self.bn2_actor_, ch.N_CU, activation_fn=None)

            # reshape the policy logits
            self.policy_logits_RB_ = tf.reshape(self.fc3_actor_RB_, (-1, 1, ch.N_CU))
            self.policy_logits_power_ = tf.reshape(self.fc3_actor_power_, (-1, 1, ch.D2D_tr_Power_levels))
             
            # self.policy_logits_ = self.combine_tensors(self.fc3_actor_power_, self.fc3_actor_RB_) # TO-DO figure out how to have 2 seperate outputs for each action type
            # might need to do two seperate updates - policy logtis for both pow_sel and RB_sel
            #print(self.policy_logits_)
            #self.policy_logits_ = tf.transpose(self.policy_logits_)
            #self.policy_logits_ = tf.expand_dims(self.policy_logits_, axis=0)
            #self.policy_logits_ = tf.reshape(self.policy_logits_, [-1, 1, action_size]) # these resize it weirdly
            # generate action probabilities for taking actions
            self.action_prob_power_ = tf.nn.softmax(self.fc3_actor_power_)
            self.action_prob_RB_ = tf.nn.softmax(self.fc3_actor_RB_)
            
      
            # set up critic network
            self.fc1_critic_ = tf.contrib.layers.fully_connected(self.input_, critic_hidden_size, activation_fn=tf.nn.elu)

            self.mean1_critic_, self.variance1_critic_ = tf.nn.moments(self.fc1_critic_, axes=-1)
            self.bn1_critic_ = tf.nn.batch_normalization(self.fc1_critic_, mean=self.mean1_critic_, variance=self.variance1_critic_, offset=0, scale=1, variance_epsilon=0.0001)

            self.fc2_critic_ = tf.contrib.layers.fully_connected(self.bn1_critic_, critic_hidden_size, activation_fn=tf.nn.elu)

            self.mean2_critic_, self.variance2_critic_ = tf.nn.moments(self.fc2_critic_, axes=-1)
            self.bn2_critic_ = tf.nn.batch_normalization(self.fc2_critic_, mean=self.mean2_critic_, variance=self.variance2_critic_, offset=0, scale=1, variance_epsilon=0.0001)

            self.baseline_ = tf.contrib.layers.fully_connected(self.bn2_critic_, 1, activation_fn=None)
      
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
            self.ac_optim_pow_ = tf.train.AdamOptimizer(learning_rate=ac_learning_rate).minimize(self.ac_loss_pow_)
            self.ac_optim_RB_ = tf.train.AdamOptimizer(learning_rate=ac_learning_rate).minimize(self.ac_loss_RB_)
            

    def get_network_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]

    def combine_tensors(self, a, b):
        c = tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1)
        c = tf.reshape(c, (-1, 10, 30))
        #c = tf.reshape(c, (-1, 2))
        return c

# hyperparameters
train_episodes = 5000  
discount = 0.99

actor_hidden_size = 32
critic_hidden_size = 32
ac_learning_rate = 0.0005
target_ac_learning_rate = 0.0005
baseline_cost = 10 #scale derivatives between actor and critic networks

# entropy hyperparameters
entropy_cost = 0.01
normalise_entropy = True

# one step returns ie TD(0).
lambda_ = 1.

action_size = ch.n_actions
obs_size = ch.N_CU

print('action_size: ', action_size)
print('obs_size: ', obs_size)

D2D_nets = []
D2D_target_nets = []
D2D_target_net_update_ops = []

tf.reset_default_graph()

for i in range(0, ch.N_D2D):
    D2D_nets.append(ActorCriticNetwork(name='ac_net_{:.0f}'.format(i), obs_size=obs_size, action_size=action_size, actor_hidden_size=actor_hidden_size,
                                       ac_learning_rate=ac_learning_rate, entropy_cost=entropy_cost, normalise_entropy=normalise_entropy,
                                       lambda_=lambda_, baseline_cost=baseline_cost))

    print('Instantiated Network {:.0f} of {:.0f}'.format(i+1, ch.N_D2D))

    D2D_target_nets.append(ActorCriticNetwork(name='ac_target_net_{:.0f}'.format(i), obs_size=obs_size, action_size=action_size, actor_hidden_size=actor_hidden_size,
                                       ac_learning_rate=target_ac_learning_rate, entropy_cost=entropy_cost, normalise_entropy=normalise_entropy,
                                       lambda_=lambda_, baseline_cost=baseline_cost))

    print('Instantiated Target Network {:.0f} of {:.0f}'.format(i+1, ch.N_D2D))

    D2D_target_net_update_ops.append(trfl.update_target_variables(D2D_target_nets[i].get_network_variables(), 
                                                                  D2D_nets[i].get_network_variables(), tau=0.001))

    print('Instantiated Target Net Update ops {:.0f} of {:.0f}'.format(i+1, ch.N_D2D))
    print('\n')


stats_rewards_list = []
stats_every = 10

initial_actions = []
#power_levels = []
#RB_selections = []

g_iB, g_j, G_ij, g_jB, G_j_j = ch.reset()
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

        for i in range(0, ch.N_D2D):
            power_action_probs.append(sess.run(D2D_nets[i].action_prob_power_, feed_dict={D2D_nets[i].input_: np.expand_dims(state,axis=0)}))
            power_action_probs[i] = power_action_probs[i][0]

            RB_action_probs.append(sess.run(D2D_nets[i].action_prob_RB_, feed_dict={D2D_nets[i].input_: np.expand_dims(state,axis=0)}))
            RB_action_probs[i] = RB_action_probs[i][0]
            
            #print('Power_action_probs: ', power_action_probs)
            #print('RB action probs: ', RB_action_probs)


            pow_sels.append(np.random.choice(ch.power_levels, p=power_action_probs[i]))
            RB_sels.append(np.random.choice(ch.CU_index, p=RB_action_probs[i]))

        #print("power_levels: ", ch.power_levels)
        CU_SINR = ch.CU_SINR_no_collision(g_iB, pow_sels, g_jB, RB_sels)
        next_state = ch.state(CU_SINR)
        D2D_SINR = ch.D2D_SINR_no_collision(pow_sels, g_j, G_ij, G_j_j, RB_sels, next_state)
        reward, net = ch.D2D_reward_no_collision(D2D_SINR, CU_SINR, RB_sels)
            
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
            bootstrap_values.append(sess.run(D2D_target_nets[i].baseline_, feed_dict={
                D2D_target_nets[i].input_: np.expand_dims(next_state, axis=0)}))
        
        #train network
        
        total_losses_pow = []
        seq_aac_returns_pow = []

        total_losses_RB = []
        seq_aac_returns_RB = []
        for i in range(0, ch.N_D2D):
            #print('Pow_sels: ',pow_sels)
            #print('RB_sels: ',RB_sels)
            #print(np.reshape([pow_sels[i], RB_sels[i]], (-1, 2)))
            #sess.run(tf.print("actions_pow", np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)), summarize=-1))

            #sess.run(tf.print("policy_logits_pow", D2D_nets[i].policy_logits_power_, summarize=-1), feed_dict={
            #    D2D_nets[i].input_: np.expand_dims(state, axis=0),
            #    D2D_nets[i].action_pow_: np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
            #    D2D_nets[i].action_RB_: np.reshape(RB_sels[i], (-1, 1)),
            #    D2D_nets[i].reward_: np.reshape(reward[i], (-1, 1)),
            #    D2D_nets[i].discount_: np.reshape(discount, (-1, 1)),
            #    D2D_nets[i].bootstrap_: np.reshape(bootstrap_values[i], (1,)) #np.expand_dims(bootstrap_value, axis=0)
            #})

            #sess.run(tf.print("inputs (state, pow_sels, RB_sels, reward, discount, bootstrap)", [np.expand_dims(state, axis=0),
            #                                                                                     np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
            #                                                                                     np.reshape(RB_sels[i], (-1, 1)),
            #                                                                                     np.reshape(reward[i], (-1, 1)),
            #                                                                                     np.reshape(discount, (-1, 1)),
            #                                                                                     np.reshape(bootstrap_values[i], (1,))]))

            _, total_loss_pow, seq_aac_return_pow = sess.run([D2D_nets[i].ac_optim_pow_, D2D_nets[i].ac_loss_pow_, D2D_nets[i].seq_aac_return_pow_], feed_dict={
                D2D_nets[i].input_: np.expand_dims(state, axis=0),
                D2D_nets[i].action_pow_: np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
                D2D_nets[i].action_RB_: np.reshape(RB_sels[i], (-1, 1)),
                D2D_nets[i].reward_: np.reshape(reward[i], (-1, 1)),
                D2D_nets[i].discount_: np.reshape(discount, (-1, 1)),
                D2D_nets[i].bootstrap_: np.reshape(bootstrap_values[i], (1,)) #np.expand_dims(bootstrap_value, axis=0)
            })
            #print("ac_optim_pow: ", _)
            #print("ac_loss_pow: ", total_loss_pow)
            #print("seq_aac_return_pow", seq_aac_return_pow)
            #print(" ")
            total_losses_pow.append(total_loss_pow)
            seq_aac_returns_pow.append(seq_aac_return_pow)
        
        #for i in range(0, ch.N_D2D):
            #print(np.reshape([pow_sels[i], RB_sels[i]], (-1, 2)))

            #sess.run(tf.print("actions_RB_before_update", np.reshape(RB_sels[i], (-1, 1)), summarize=-1))

            #sess.run(tf.print("inputs (state, pow_sels, RB_sels, reward, discount, bootstrap)", [np.expand_dims(state, axis=0),
            #                                                                                     np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
            #                                                                                     np.reshape(RB_sels[i], (-1, 1)),
            #                                                                                     np.reshape(reward[i], (-1, 1)),
            #                                                                                     np.reshape(discount, (-1, 1)),
            #                                                                                     np.reshape(bootstrap_values[i], (1,))]))

            #sess.run(tf.print("policy_logits_RB_before_update", D2D_nets[i].policy_logits_RB_, summarize=-1), feed_dict={
            #    D2D_nets[i].input_: np.expand_dims(state, axis=0),
            #    D2D_nets[i].action_pow_: np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
            #    D2D_nets[i].action_RB_: np.reshape(RB_sels[i], (-1, 1)),
            #    D2D_nets[i].reward_: np.reshape(reward[i], (-1, 1)),
            #    D2D_nets[i].discount_: np.reshape(discount, (-1, 1)),
            #    D2D_nets[i].bootstrap_: np.reshape(bootstrap_values[i], (1,)) #np.expand_dims(bootstrap_value, axis=0)
            #})
        
        
        
        for i in range(0, ch.N_D2D):
            #print(np.reshape([pow_sels[i], RB_sels[i]], (-1, 2)))

            #sess.run(tf.print("actions_RB", np.reshape(RB_sels[i], (-1, 1)), summarize=-1))

            #sess.run(tf.print("policy_logits_RB", D2D_nets[i].policy_logits_RB_, summarize=-1), feed_dict={
            #    D2D_nets[i].input_: np.expand_dims(state, axis=0),
            #    D2D_nets[i].action_pow_: np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
            #    D2D_nets[i].action_RB_: np.reshape(RB_sels[i], (-1, 1)),
            #    D2D_nets[i].reward_: np.reshape(reward[i], (-1, 1)),
            #    D2D_nets[i].discount_: np.reshape(discount, (-1, 1)),
            #    D2D_nets[i].bootstrap_: np.reshape(bootstrap_values[i], (1,)) #np.expand_dims(bootstrap_value, axis=0)
            #})

            _, total_loss_RB, seq_aac_return_RB = sess.run([D2D_nets[i].ac_optim_RB_, D2D_nets[i].ac_loss_RB_, D2D_nets[i].seq_aac_return_RB_], feed_dict={
                D2D_nets[i].input_: np.expand_dims(state, axis=0),
                D2D_nets[i].action_pow_: np.reshape(np.where(ch.power_levels == pow_sels[i]), (-1, 1)),
                D2D_nets[i].action_RB_: np.reshape(RB_sels[i], (-1, 1)),
                D2D_nets[i].reward_: np.reshape(reward[i], (-1, 1)),
                D2D_nets[i].discount_: np.reshape(discount, (-1, 1)),
                D2D_nets[i].bootstrap_: np.reshape(bootstrap_values[i], (1,)) #np.expand_dims(bootstrap_value, axis=0)
            })
            #print("ac_optim_RB: ", _)
            #print("ac_loss_RB_: ", total_loss_RB)
            #print("seq_aac_return_RB_", seq_aac_return_RB)
            #print(" ")
            total_losses_RB.append(total_loss_RB)
            seq_aac_returns_RB.append(seq_aac_return_RB)

        total_loss_list_pow.append(np.mean(total_losses_pow))
        total_loss_list_RB.append(np.mean(total_losses_RB))
        
        #update target network
        for i in range(0, ch.N_D2D):
            sess.run(D2D_target_net_update_ops[i])
        
        #some useful things for debuggin
        stats_actor_loss_pow += np.mean(seq_aac_return_RB.extra.policy_gradient_loss)
        stats_critic_loss_pow += np.mean(seq_aac_return_RB.extra.baseline_loss)

        stats_actor_loss_RB += np.mean(seq_aac_return_pow.extra.policy_gradient_loss)
        stats_critic_loss_RB += np.mean(seq_aac_return_pow.extra.baseline_loss)
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

    true_collisions_fig = plt.figure()
    plt.plot(eps, collisions)
    plt.ylabel('Number of collisions')
    plt.xlabel('Time-slot')
    plt.show()

    collision_prob_fig = plt.figure()
    plt.plot(eps[-len(smoothed_col_probs):], smoothed_col_probs)
    plt.plot(eps, D2D_collision_probs, color='grey', alpha=0.3)
    plt.xlabel('Time-slot')
    plt.ylabel('D2D collision probability')
    plt.show()

    access_rate_fig = plt.figure()
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
    

