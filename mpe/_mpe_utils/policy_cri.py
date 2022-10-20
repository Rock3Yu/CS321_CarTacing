from email import policy
from random import random
import numpy as np
from pyglet.window import key

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 1.0 #left
            if self.move[1]: u[2] += 1.0 #right
            if self.move[2]: u[4] += 1.0 #down
            if self.move[3]: u[3] += 1.0 #up
            if True not in self.move:
                u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True
    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False
class TestPolicy(Policy):
    def action(self,observation,reward): #receive observation,reward,done,info
        x=-1
        y=-1
        return np.array([0.,max(x,0.),max(-x,0.),max(y,0.),max(-y,0.)]) 
        my_vel=observation[0][0]
        my_pos=observation[0][1]
        cri_vel=observation[1][0]#adversary is front of the agents
        cri_pos=observation[2][0]
        partner_vel=observation[1][1:]#other policy
        partner_pos=observation[2][1:]
        landmarks=observation[1] #landmark position

