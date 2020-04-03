import gym
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
import time


if __name__=='__main__':
    env = gym.make('FrozenLake-v0')
    agent = Agent(step_size=0.001, discount_factor=0.9995, num_actions=4, num_states=16,
    epsilon_start=1.0, epsilon_max=1.0, epsilon_dec=0.9999995, epsilon_min=0.01)
    scores=[]
    win_pct_list=[]
    n_games=500000

    for i in range(n_games):
        done =False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_,reward, done, info =env.step(action)
            agent.learn(observation, action, reward, observation_)
            score+=reward
            observation=observation_
        scores.append(score)
        if i % 100 ==0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if i%1000==0:
                print(f'episode:{i}, win_pct:{win_pct:.2f}, epsilon:{agent.epsilon:.2f}')

        if i%50000==0:
            time.sleep(0.01)
            env.render()

    plt.plot(win_pct_list)
    plt.show()
