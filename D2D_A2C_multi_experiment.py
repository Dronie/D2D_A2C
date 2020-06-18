import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import trfl
import matplotlib.pyplot as plt
import n_player_game as game

writer = tf.summary.FileWriter("/home/stefan/tmp/D2D/2")

env = game.Game(no_players=25, no_counters=30)

# set up Actor and Critic networks
class ActorCriticNetwork:
    def __init__(self, name, obs_size=2, action_size=1, actor_hidden_size=32, critic_hidden_size=32, ac_learning_rate=0.001,  
                 entropy_cost=0.01, normalise_entropy=True, lambda_=0., baseline_cost=1.):
    
        with tf.variable_scope(name):
            # network variables
            self.name=name
            self.input_ = tf.placeholder(tf.float32, [None, obs_size], name='inputs')
            self.action_ = tf.placeholder(tf.int32, [None, 1], name='action')
            self.reward_ = tf.placeholder(tf.float32, [None, 1], name='reward')
            self.discount_ = tf.placeholder(tf.float32, [None, 1], name='discount')
            self.bootstrap_ = tf.placeholder(tf.float32, [None], name='bootstrap')

            # set up actor network (approximates optimal policy)
            self.fc1_actor_ = tf.contrib.layers.fully_connected(self.input_, actor_hidden_size, activation_fn=tf.nn.elu)
            self.fc2_actor_ = tf.contrib.layers.fully_connected(self.fc1_actor_, actor_hidden_size, activation_fn=tf.nn.elu)
            self.fc3_actor_ = tf.contrib.layers.fully_connected(self.fc2_actor_, action_size, activation_fn=None)
            # reshape the policy logits
            self.policy_logits_ = tf.reshape(self.fc3_actor_, [-1, 1, action_size] )
  
            # generate action probabilities for taking actions
            self.action_prob_ = tf.nn.softmax(self.fc3_actor_)
      
            # set up critic network (approximates optimal state-value function (used as a baseline to reduce variance of loss gradient))
            # - uses policy evaluation (e.g. Monte-Carlo / TD learning) to estimate the advantage
            self.fc1_critic_ = tf.contrib.layers.fully_connected(self.input_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.fc2_critic_ = tf.contrib.layers.fully_connected(self.fc1_critic_, critic_hidden_size, activation_fn=tf.nn.elu)
            self.baseline_ = tf.contrib.layers.fully_connected(self.fc2_critic_, 1, activation_fn=None)
      
            # Calculates the loss for an A2C update along a batch of trajectories. (TRFL)
            self.seq_aac_return_ = trfl.sequence_advantage_actor_critic_loss(self.policy_logits_, self.baseline_, self.action_,
               self.reward_, self.discount_, self.bootstrap_, lambda_=lambda_, entropy_cost=entropy_cost, 
               baseline_cost=baseline_cost, normalise_entropy=normalise_entropy)
      
            # Optimize the loss
            self.ac_loss_ = tf.reduce_mean(self.seq_aac_return_.loss)
            self.ac_optim_ = tf.train.AdamOptimizer(learning_rate=ac_learning_rate).minimize(self.ac_loss_)
      
    def get_network_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]

# hyperparameters
max_timesteps = 1000  
discount = 0.99

actor_hidden_size = 32 # number of units per layer in actor net
critic_hidden_size = 32 # number of units per layer in critic net

ac_learning_rate = 0.005

baseline_cost = 10. #scale derivatives between actor and critic networks
#The `baseline_cost` parameter scales the
#  gradients w.r.t the baseline relative to the policy gradient. i.e:
#  `d(loss) / d(baseline) = baseline_cost * (n_step_return - baseline)`.

# entropy hyperparameters
entropy_cost = 0.001
normalise_entropy = True

# lambda_: an optional scalar or 2-D Tensor with shape `[T, B]` for
#        Generalised Advantage Estimation as per
#        https://arxiv.org/abs/1506.02438.
lambda_ = 0.

action_size = env.no_counters
obs_size = env.no_counters

print('action_size: ', action_size)
print('obs_size: ', obs_size)

player_nets = []
player_target_nets = []
player_target_net_update_ops = []

tf.reset_default_graph()

for i in range(0, env.no_players):
    player_nets.append(ActorCriticNetwork(name='ac_net_{:.0f}'.format(i), obs_size=obs_size, action_size=action_size, actor_hidden_size=actor_hidden_size,
                                       ac_learning_rate=ac_learning_rate, entropy_cost=entropy_cost, normalise_entropy=normalise_entropy,
                                       lambda_=lambda_, baseline_cost=baseline_cost))

    print('Instantiated Network {:.0f} of {:.0f}'.format(i+1, env.no_players))

    player_target_nets.append(ActorCriticNetwork(name='ac_target_net_{:.0f}'.format(i), obs_size=obs_size, action_size=action_size, actor_hidden_size=actor_hidden_size,
                                       ac_learning_rate=ac_learning_rate, entropy_cost=entropy_cost, normalise_entropy=normalise_entropy,
                                       lambda_=lambda_, baseline_cost=baseline_cost))

    print('Instantiated Target Network {:.0f} of {:.0f}'.format(i+1, env.no_players))

    player_target_net_update_ops.append(trfl.update_target_variables(player_target_nets[i].get_network_variables(), 
                                                                  player_nets[i].get_network_variables(), tau=0.001))

    print('Instantiated Target Net Update ops {:.0f} of {:.0f}'.format(i+1, env.no_players))
    print('\n')


stats_rewards_list = []

RB_selections = []
    
state =  env.initialize()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    total_reward, t_length, done = 0, 0, 0
    stats_actor_loss, stats_critic_loss = 0., 0.
    total_loss_list, action_list, action_prob_list, bootstrap_list = [], [], [], []
    rewards_list = []
    collision_var = 0

    # used for plots
    D2D_collision_probs = []
    collisions = []
    access_ratios = []
    access_rates = []
    avg_throughput = []
    time_avg_throughput = []

    total_rewards = []

    for t in range(1, max_timesteps):
        env.reset()
        #ch.collision_indicator = 0
                     
        # generate action probabilities from policy net and sample from the action probs (policy)
        action_probs = []
        actions = []
        power_levels = []
        RB_selections = []

        for i in range(0, env.no_players):
            action_probs.append(sess.run(player_nets[i].action_prob_, feed_dict={player_nets[i].input_: np.expand_dims(state,axis=0)}))
            action_probs[i] = action_probs[i][0] 
            actions.append(np.random.choice(np.arange(len(action_probs[i])), p=action_probs[i])) # has a problem with this on the 2nd iteration ('float object cannot be interpreted as an integer')
        
        print(actions)

        next_state = env.choose_counters(actions)

        rewards = []

        for i in range(0, env.no_players):
            rewards.append(env.get_reward(next_state))
            
        total_reward = sum(rewards)
        print('Total Reward: ', total_reward)
        total_rewards.append(total_reward)

        #if ch.collision_indicator > 0:
        #    collision_var += 1
        
        #collisions.append(ch.collision_indicator)
        
        D2D_collision_probs.append(collision_var / t)

        #next_state = np.clip(next_state,-1.,1.)

        t_length += 1

        # bootstrapping: stopping to update many times in a trajectory
        #  - done by estimating the value of the current state using the critic net
        #
        # sutton - "updating the value estimate for a state
        #           from the estimated values of subsequent states" - given by critic net
        #
        #           "Only through bootstrapping do we introduce bias and 
        #            an asymptotic dependance on the quality of the
        #            function approximation" - this bias "often reduces variance
        #                                      and accelerates learning"

        # bootstrapping procedures (need to fix - bootstrapping happens once every timestep)
        if t == max_timesteps:
          bootstrap_value = np.zeros((1,),dtype=np.float32)
        else:
          #get bootstrap values
          bootstrap_values = []
          for i in range(0, env.no_players):
            bootstrap_values.append(sess.run(player_target_nets[i].baseline_, feed_dict={
                player_target_nets[i].input_: np.expand_dims(next_state, axis=0)}))
        
        # update network parameters
        total_losses = []
        seq_aac_returns = []

        for i in range(0, env.no_players):
            sess.run(tf.print("actions", np.reshape(actions[i], (-1, 1)), summarize=-1))

            sess.run(tf.print("policy_logits", player_nets[i].policy_logits_, summarize=-1), feed_dict={
                player_nets[i].input_: np.expand_dims(state, axis=0),
                player_nets[i].action_: np.reshape(actions[i], (-1, 1)),
                player_nets[i].reward_: np.reshape(rewards[i], (-1, 1)),
                player_nets[i].discount_: np.reshape(discount, (-1, 1)),
                player_nets[i].bootstrap_: np.reshape(bootstrap_values[i], (1,)) #np.expand_dims(bootstrap_value, axis=0)
            })

            _, total_loss, seq_aac_return = sess.run([player_nets[i].ac_optim_, player_nets[i].ac_loss_, player_nets[i].seq_aac_return_], feed_dict={
                player_nets[i].input_: np.expand_dims(state, axis=0),
                player_nets[i].action_: np.reshape(actions[i], (-1, 1)),
                player_nets[i].reward_: np.reshape(rewards[i], (-1, 1)),
                player_nets[i].discount_: np.reshape(discount, (-1, 1)),
                player_nets[i].bootstrap_: np.reshape(bootstrap_values[i], (1,))
            })
            total_losses.append(total_loss)
            seq_aac_returns.append(seq_aac_return)

        total_loss_list.append(np.mean(total_losses))

        
        # update target network
        for i in range(0, env.no_players):
            sess.run(player_target_net_update_ops[i])
        
        if t % 10 == 0:
            print(env.counters)


        state = next_state

        print(t)

        #writer.add_graph(sess.graph)
        #print("Graph added!")

    
    stats_rewards_list.append((t, total_reward, t_length))
    ts, rews, lens = np.array(stats_rewards_list).T

    ts = np.arange(999)
    smoothed_rews = running_mean(total_rewards, 10)

    reward_fig = plt.figure()

    plt.plot(ts[-len(smoothed_rews):], smoothed_rews)
    plt.plot(ts, total_rewards, color='grey', alpha=0.3)
    plt.xlabel('Time-slot')
    plt.ylabel('Reward')
    plt.show()







