import random


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.Q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
 
    
    def update_q(self, state, action, reward, next_state):
        max_next_q = max([self.get_q(next_state, a) for a in self.actions])
        q_realistic = reward + self.gamma * max_next_q

        q_estimate = self.Q.get((state, action), None)
        if q_estimate is None:
            self.Q[(state, action)] = reward
        else:
            self.Q[(state, action)] = (q_estimate + self.alpha * (q_realistic - q_estimate))


    def choose_action(self, state, training=True):
        q_actions = [self.get_q(state, a) for a in self.actions]
        action = self.actions[q_actions.index(max(q_actions))]
        if training:
            if random.random() < self.epsilon:
                action = random.choice(self.actions)
        return action


    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)
# end class