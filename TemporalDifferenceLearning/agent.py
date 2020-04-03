import numpy as np

class Agent():

    def __init__(self, num_states, num_actions, step_size, discount_factor,
    epsilon_max, epsilon_min, epsilon_dec, epsilon_start):
        self.num_states = num_states
        self.num_actions = num_actions
        self.step_size = step_size
        self.discount_factor = discount_factor
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.epsilon = epsilon_start

        self.Q={}

        self.init_Q()

    def init_Q(self):
        for state in range(self.num_states):
            for action in range(self.num_actions):
                self.Q[(state, action)] = 0.0


    def choose_action(self, state):

        if np.random.random()<self.epsilon:
            action = np.random.choice([i for i in range(self.num_actions)])
        else:
            actions = np.array([self.Q[(state,a)] for a in range(self.num_actions)])
            action = np.argmax(actions)
        return action

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_dec)


    def learn(self, state, action, reward, state_):
        actions = np.array([self.Q[(state, a)] for a in range(self.num_actions)])
        a_max = np.argmax(actions)

        self.Q[(state, action)] += self.step_size*(reward+self.discount_factor*self.Q[(state_,a_max)] - self.Q[(state, action)])

        self.decrement_epsilon()
