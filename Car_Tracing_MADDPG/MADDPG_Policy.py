import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

import Car_Tracing_Scenario

class MADDPG_Policy:
    def __init__(self, num_police, num_criminal):
        self.num_police = num_police
        self.num_criminal = num_criminal
        self.num_agent = num_police + num_criminal
        self.actors = [None] * self.num_agent
        self.actors_target = [None] * self.num_agent
        self.critics = [None] * self.num_agent
        self.critics_target = [None] * self.num_agent
        self.optimizer_critics = [None] * self.num_agent
        self.optimizer_actors = [None] * self.num_agent
        self.memory = self.Memory(int(1e3))
        self.tao = .01
        self.gamma = .97

        for i in range(self.num_agent):
            self.actors[i] = self.Actor()
            self.actors_target[i] = self.Actor()
            self.critics[i] = self.Critic()
            self.critics_target[i] = self.Critic()
            self.optimizer_actors[i] = optim.Adam(self.actors[i].parameters(), lr=1.)
            self.optimizer_critics[i] = optim.Adam(self.critics[i].parameters(), lr=1.)
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

    def act(self, num_epoch, cycles, train_batch):
        #env = Car_Tracing_Scenario.env(map_path = 'map/test_map_d.txt', max_cycles = cycles, ratio = .9)#4
        #env = Car_Tracing_Scenario.env(map_path = 'map/test_map.txt', max_cycles = cycles, ratio = 1.5)#4
        env = Car_Tracing_Scenario.env(map_path = 'map/city.txt', max_cycles = cycles, ratio = .9)#4
        self.load()
        for epoch in range(num_epoch):
            env.reset()
            obs = env.last()[0]['obs']
            for step in range(cycles):
                if (step+1) % 50 == 0:
                    print(f'training...  step={step+1}/{cycles}')
                actions = []
                rewards = []
                for i in range(self.num_agent):
                    env.render()
                    action = self.actors[i](obs).detach().numpy()
                    action /= np.max(np.abs(action))
                    actions.append(action)
                    env.step(action)
                    rewards.append(env.rewards[env.last()[0]['rew']])
                obs_new = env.last()[0]['obs']
                self.memory.add(obs, actions, rewards, obs_new)
                obs = obs_new
                if step > 3:
                    for cnt in range(train_batch):
                        for i in range(self.num_agent):
                            obs, actions, rewards, obs_new = self.memory.sample()
                            actions_new = []
                            for j in range(self.num_agent):
                                action_new = self.actors_target[j](obs_new).detach().numpy()
                                action_new /= np.max(np.abs(action_new))
                                actions_new.append(action_new)
                            Q = self.critics[i](obs, actions)
                            Q_target = self.critics_target[i](obs_new, actions_new)
                            y = torch.tensor(rewards[i]) + self.gamma * Q_target
                            self.optimizer_critics[i].zero_grad()
                            loss_critic = nn.MSELoss()(Q, y)
                            loss_critic.backward()
                            self.optimizer_critics[i].step()
                            self.optimizer_actors[i].zero_grad()
                            loss_actor = -torch.mean(self.critics[i](obs, actions))
                            loss_actor.backward()
                            self.optimizer_actors[i].step()
                    for i in range(self.num_agent):
                        cur = self.critics[i].state_dict()
                        tar = self.critics_target[i].state_dict()
                        for key in cur.keys():
                            tar[key] = self.tao * cur[key] + (1 - self.tao) * tar[key]
                        self.critics_target[i].load_state_dict(tar)
                        cur = self.actors[i].state_dict()
                        tar = self.actors_target[i].state_dict()
                        for key in cur.keys():
                            tar[key] = self.tao * cur[key] + (1 - self.tao) * tar[key]
                        self.actors_target[i].load_state_dict(tar)
            self.save()

    def save(self):
        for i in range(self.num_agent):
            print(f'saving...  idx={i}')
            torch.save(self.actors[i].state_dict(), f'./data/actors/{i}')
            torch.save(self.actors_target[i].state_dict(), f'./data/actors_target/{i}')
            torch.save(self.critics[i].state_dict(), f'./data/critics/{i}')
            torch.save(self.critics_target[i].state_dict(), f'./data/critics_target/{i}')

    def load(self):
        for i in range(self.num_agent):
            print(f'loading...  idx={i}')
            self.actors[i].load_state_dict(torch.load(f'./data/actors/{i}'))
            self.actors_target[i].load_state_dict(torch.load(f'./data/actors_target/{i}'))
            self.critics[i].load_state_dict(torch.load(f'./data/critics/{i}'))
            self.critics_target[i].load_state_dict(torch.load(f'./data/critics_target/{i}'))

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.temp = nn.Linear(3478 * 2, 2)

        def forward(self, obs):
            return self.temp(torch.tensor(obs, dtype=torch.float).flatten())

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.temp1 = nn.Linear(3478 * 2, 1)
            self.temp2 = nn.Linear(4 * 2, 1)
            self.temp3 = nn.Linear(2, 1)

        def forward(self, obs, action):
            t1 = self.temp1(torch.tensor(obs, dtype=torch.float).flatten())
            t2 = self.temp2(torch.tensor(action, dtype=torch.float).flatten())
            return self.temp3(torch.cat([t1, t2]))

    class Memory:
        def __init__(self, size):
            self.size = size
            self.buffer_obs = [None] * size
            self.buffer_actions = [None] * size
            self.buffer_rewards = [None] * size
            self.buffer_obs_new = [None] * size
            self.cur_idx = 0
            self.total_cnt = 0

        def add(self, obs, actions, rewards, obs_new):
            self.buffer_obs[self.cur_idx] = obs
            self.buffer_actions[self.cur_idx] = actions
            self.buffer_rewards[self.cur_idx] = rewards
            self.buffer_obs_new[self.cur_idx] = obs_new
            self.cur_idx = (self.cur_idx + 1) % self.size
            self.total_cnt += 1

        def sample(self):
            if self.total_cnt < self.size:
                idx = random.randrange(0, self.total_cnt)
            else:
                idx = random.randrange(0, self.size)
            return self.buffer_obs[idx], self.buffer_actions[idx], self.buffer_rewards[idx], self.buffer_obs_new[idx]

if __name__ == '__main__':
    policy = MADDPG_Policy(3, 1)
    policy.act(5, 250, 5)