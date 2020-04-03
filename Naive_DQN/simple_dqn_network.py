import gym
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from utils import plot_learning_curve


class LinearDeepQNetwork(nn.Module):

    def __init__(self, lr, n_actions, input_dims):
        super().__init__()
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(in_features=128, out_features=n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)  # send sself to device

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions


class Agent():

    def __init__(self, lr, input_dims, n_actions, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_dec=1e-5):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.action_space = [i for i in range(n_actions)]
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()

        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - \
            self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]
        q_target = reward + self.gamma * self.Q.forward(states_).max()
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)

        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []
    agent = Agent( lr = 0.0001, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n)
    for i in range(n_games):

        done = False
        obs = env.reset()
        score=0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_

            if i % 900 == 0 and False:
                time.sleep(0.05)
                env.render()
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'episode:{i}, score:{score:.2f}, epsilon:{agent.epsilon :.2f}')

    filename = 'cartpole_naive_dqn.png'

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x,scores, eps_history, filename)
