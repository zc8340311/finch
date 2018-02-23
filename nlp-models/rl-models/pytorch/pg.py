import torch
import numpy as np


class PolicyGradient(torch.nn.Module):
    def __init__(self, env, n_in, n_hidden, n_out, lr=0.01):
        super(PolicyGradient, self).__init__()
        self.env = env
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr
        self.build_model()
    # end constructor


    def build_model(self):
        self.hidden_net = torch.nn.Sequential(*self._hidden_net())
        self.logits = torch.nn.Linear(self.n_hidden[-1], self.n_out)
        self.softmax = torch.nn.Softmax()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
    # end method


    def _hidden_net(self):
        hidden_net = []
        forward = [self.n_in] + self.n_hidden
        for i in range(1, len(forward)):
            hidden_net.append(torch.nn.Linear(forward[i-1], forward[i]))
            hidden_net.append(torch.nn.ELU())
        return hidden_net
    # end method


    def forward(self, inputs):
        logits = self.logits(self.hidden_net(inputs))
        probas = self.softmax(logits)
        action = torch.autograd.Variable(torch.multinomial(probas, num_samples=1).data, requires_grad=False)
        return logits, action
    # end method


    def get_gradients(self, obs):
        logits, action = self.forward(obs)
        loss = self.criterion(logits, torch.squeeze(action, 1))
        self.optimizer.zero_grad()
        loss.backward()
        gradients = [param.grad.data.numpy() for param in self.parameters()]
        return action, gradients


    def learn(self, n_games_per_update=10, n_max_steps=1000, n_iterations=250, discount_rate=0.95):
        for iteration in range(n_iterations):
            print("Iteration: {}".format(iteration))
            ep_rewards = []            # rewards in one eposide
            ep_gradients = []          # gradients in one eposide
            for game in range(n_games_per_update):
                game_rewards = []      # rewards in one round of game
                game_gradients = []    # gradients in one round of game
                obs = self.env.reset()
                for step in range(n_max_steps):
                    obs = torch.autograd.Variable(torch.from_numpy(np.atleast_2d(obs).astype(np.float32)))
                    action_val, gradients_val = self.get_gradients(obs)
                    obs, reward, done, info = self.env.step(action_val.data.numpy()[0][0])
                    game_rewards.append(reward)
                    game_gradients.append(gradients_val)
                    if done:
                        break
                ep_rewards.append(game_rewards)
                ep_gradients.append(game_gradients)

            self.optimizer.zero_grad()
            ep_rewards = self.discount_and_normalize_rewards(ep_rewards, discount_rate)
            for var_idx, param in enumerate(self.parameters()):
                mean_gradients = np.mean([reward * ep_gradients[game_idx][step_idx][var_idx]
                                          for game_idx, game_rewards in enumerate(ep_rewards)
                                          for step_idx, reward in enumerate(game_rewards)], axis=0)
                param.grad = torch.autograd.Variable(torch.from_numpy(mean_gradients.astype(np.float32)))
            self.optimizer.step()
    # end method


    def play(self):
        obs = self.env.reset()
        done = False
        count = 0
        while not done:
            self.env.render()
            obs = torch.autograd.Variable(torch.from_numpy(np.atleast_2d(obs).astype(np.float32)))
            _, action_val = self.forward(obs)
            obs, reward, done, info = self.env.step(action_val.data.numpy()[0][0])
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
