from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch.nn as nn
import torch.nn.functional as F
from  torch.nn.modules.loss import _Loss
from typing import Callable, Optional
from torch import Tensor

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, config=None, model=None):
        device = "cuda"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if config is not None :
            self.nb_actions = config['nb_actions']
            self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
            self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
            buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
            self.memory = ReplayBuffer(buffer_size,device)
            self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
            self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
            self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
            self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
            self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
            self.model = model.to(device)
            self.target_model = deepcopy(self.model).to(device)
            self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
            lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
            self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
            self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
            self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
            self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
            self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        else :
            state_dim = 6
            n_action = 4
            nb_neurons=512
            DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                                      nn.ReLU(),
                                      nn.Linear(nb_neurons, nb_neurons),
                                      nn.ReLU(), 
                                      nn.Linear(nb_neurons, nb_neurons),
                                      nn.ReLU(), 
                                      nn.Linear(nb_neurons, n_action)).to(self.device)

            self.model = DQN
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        max_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.3f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                if episode_cum_reward >= max_reward :
                    max_reward = episode_cum_reward
                    self.save(path="model_"+str(int(10*episode_cum_reward))+".pth")
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return
    def greedy_action(self, state):
        device = "cuda"
        with torch.no_grad():
            Q = self.model(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
    def act(self, observation, use_random=False):
        if use_random:
            a = self.env.action_space.sample()
        else :
            a = self.greedy_action(observation)
        return a
    def save(self, path):
        path = self.path if None else path
        print(f"Sauvegarde du modèle à : {path}")
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load('model.pth', map_location=self.device))