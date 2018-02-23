import tensorflow as tf
import numpy as np


class PolicyGradient:
    def __init__(self, env, n_in, hidden_net, n_out, lr=0.01, sess=tf.Session()):
        self.env = env
        self.n_in = n_in
        self.hidden_net = hidden_net
        self.n_out = n_out
        self.lr = lr
        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_forward_path()
        self.add_backward_path()
    # end method


    def add_input_layer(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_in])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.actions = tf.placeholder(tf.int32, shape=[None])
    # end method


    def add_forward_path(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_in])
        hidden = self.hidden_net(self.X)
        self.logits = tf.layers.dense(hidden, self.n_out)
        self.action = tf.multinomial(self.logits, num_samples=1)
    # end method


    def add_backward_path(self):
        # maximize (log_p * R) = minimize -(log_p * R)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.logits)
        self.loss = tf.reduce_sum(xentropy * self.rewards) # reward guided loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method


    def learn(self, n_games_per_update=10, n_max_steps=1000, n_iterations=500, discount_rate=0.95):
        self.sess.run(tf.global_variables_initializer())

        for iteration in range(n_iterations):
            ep_rewards = [] # rewards in one eposide
            ep_obs = []
            ep_actions = []

            for game in range(n_games_per_update):
                game_rewards = [] # rewards in one round of game
                obs = self.env.reset()
                for step in range(n_max_steps):
                    action_val = self.sess.run(self.action, {self.X: np.atleast_2d(obs)})
                    obs, reward, done, info = self.env.step(action_val[0][0])
                    ep_obs.append(obs)
                    ep_actions.append(action_val[0][0])
                    game_rewards.append(reward)
                    if done:
                        break
                ep_rewards.append(game_rewards)

            ep_rewards = self.discount_and_normalize_rewards(ep_rewards, discount_rate)
            flat_rewards = np.concatenate(ep_rewards)
            _, loss = self.sess.run([self.train_op, self.loss], {self.X: np.vstack(ep_obs),
                                                                 self.rewards: flat_rewards,
                                                                 self.actions: np.array(ep_actions)})
            print("Iteration: [%d / %d], Loss: %.3f" % (iteration, n_iterations, loss))
    # end method


    def play(self):
        obs = self.env.reset()
        done = False
        count = 0
        while not done:
            self.env.render()
            action_val = self.sess.run(self.action, {self.X: np.atleast_2d(obs)})
            obs, reward, done, info = self.env.step(action_val[0][0])
            count += 1
        print(count)
    # end method


    def discount_rewards(self, game_rewards, discount_rate):
        discounted_rewards = np.zeros(len(game_rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(game_rewards))):
            cumulative_rewards = game_rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards
    # end method


    def discount_and_normalize_rewards(self, ep_rewards, discount_rate):
        discounted = [self.discount_rewards(game_rewards, discount_rate) for game_rewards in ep_rewards]
        flat_rewards = np.concatenate(discounted)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(game_rewards - reward_mean) / reward_std for game_rewards in discounted]
    # end method
# end class
