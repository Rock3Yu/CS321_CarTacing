import random
import sys
from tkinter import E
from turtle import color
#absolute import, python will find the top package in sys.path, we show subpackage by '.'
sys.path.append("D:\\Source\\Project_Big\\model\\PettingZoo")
import numpy as np
from gym.utils.ezpickle import EzPickle

from pettingzoo_pc.utils.conversions import parallel_wrapper_fn

from pettingzoo_pc.mpe._mpe_utils.core import Agent, Landmark, World,Entity
from pettingzoo_pc.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo_pc.mpe._mpe_utils.simple_env_mine import SimpleEnv, make_env
from pettingzoo_pc.mpe._mpe_utils import simple_env_mine

filename = 'D:/Source/project_Big/model/PettingZoo/pettingzoo_pc/mpe/_mpe_utils/maps/nobound_withland.txt'
max_cycle=1200

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2, #障碍物在map中识别，
        max_cycles=100,
        continuous_actions=True,
    ):
        EzPickle.__init__(
            self,
            num_good,
            num_adversaries,
            num_obstacles,
            max_cycles,
            continuous_actions,
        )
        scenario = Scenario()
        first=exchange()
        map,num_good,num_adversaries,num_obstacles=first.generator(filename,max_cycles)
        world = scenario.make_world(num_good, num_adversaries, num_obstacles,map)
        super().__init__(scenario, world, max_cycles, continuous_actions,None,map,num_adversaries)
        #在上面，已经加载了pc的scenario，pc.scenario的makeworld，在init中进行了reset_world(of pc.scenrio)
        self.metadata["name"] = "police_criminal"


env = make_env(raw_env)  #wrap the env
parallel_env = parallel_wrapper_fn(env)

class My_World(World):  # multi-agent world
    def __init__(self):
        super().__init__()
        self.endpoints=[]


class Scenario(BaseScenario):
    def make_world(self, num_good=3, num_adversaries=1, num_obstacles=2,map=[]):
        world = My_World()
        # set any world properties first
        world.dim_c = 0
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False #前i个agent表示criminal即 adversay
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = False #cannot send communication signals
            agent.size = 0.048 if agent.adversary else 0.048
            agent.accel = 0.4 if agent.adversary else 0.3 #accelarate
            agent.max_speed = 0.52 if agent.adversary else 0.4
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.016
            landmark.boundary = False #边界
        return world

    def reset_world(self, world, np_random,map):
        # print(map)
        landmarks=[]
        polices=[]
        criminals=[]
        ends=[]
        for i,row in enumerate(map):
            for j,el in enumerate(row):
                if el=='-':
                    pass
                if el == '1' :
                    landmarks.append((i,j))
                elif el=='p':
                    polices.append((i,j))
                elif el=='c':
                    criminals.append((i,j))
                elif el=='e':
                    ends.append((i,j))

        l_size = world.landmarks[0].size if len(world.landmarks)>0 else  0.016
        for end in ends:
            a= Entity()
            a.collide=False
            a.size=0.04
            a.color=np.array([0.35, 0.35, 0.85]) #blue
            a.state.p_pos=np.array([end[1]*l_size*2.,end[0]*l_size*2.],dtype='float')
            world.endpoints.append(a)
        # print(landmarks,polices,criminals)
        # load properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.85, 0.35, 0.35])
                if not agent.adversary
                else np.array([0.35, 0.85, 0.35]) #criminal green
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            if not landmark.boundary:
                landmark.state.p_pos = np.array([landmarks[i][1]*l_size*2.,landmarks[i][0]*l_size*2.],dtype='float')
                landmark.state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        num_adversaries=len(criminals)
        for i,agent in enumerate(world.agents):
            agent.state.p_pos = np.array([criminals[i][1]*l_size*2.
            ,criminals[i][0]*l_size*2.] if i < num_adversaries else [polices[i-num_adversaries][1]*l_size*2.,polices[i-num_adversaries][0]*l_size*2.],dtype='float')
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        # num_cri=0
        for i,other in enumerate(world.agents): #其他所有的智体包括adversary and self
            # if other is agent:
            #     continue
            comm.append(other.state.c)
            # if other.adversary==True:
            #     num_cri=i
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # if not other.adversary: 因为要实现全局观察因此去掉这个限制
            other_vel.append(other.state.p_vel)
        return np.array([[agent.state.p_vel,agent.state.p_pos],other_vel,other_pos,entity_pos],dtype=object)
        # return np.concatenate(
        #     [agent.state.p_vel]
        #     + [agent.state.p_pos]
        #     + entity_pos
        #     + other_pos
        #     + other_vel
        # )



class exchange:
    '''0 is empty, 1 stands for solid obstacles, 2 stands for hinder zone, 3 stands for movable obstacles,
    e stands for endpoints for criminal
    start points: p stands for policy car, c stands for policy criminal car. 
    '''
    def make_map(filename,height=125,width=100):
        file=open(filename,'w')
        array=np.zeros((height,width),dtype='int')
        array[0,:]='1'
        array[array.shape[0]-1,:]='1'
        array[:,0]='1'
        array[:,array.shape[1]-1]='1'
        # array=''.join(array)
        str_a=''
        for row in array:
            for i in row:
                str_a=str_a+(str(i))
            str_a=str_a+('\n')
        print(str_a)
        file.write(str_a)


    # def origin_env_start():
    #     env=simple_env_mine.env()
    #     env.reset()
    #     for agent in env.agent_iter(1):
    #         env.render()
    #     # input("Press Enter to continue...")
    #         observation, reward, done, info = env.last()
    #     #action = policy.action(observation)
    #         action = []
    #         env.step(action)   
    #     env.close()
    def generator(self,filename,max_cycles): #地图两个智体的数目,总回合数
        file = open(filename,'r') # 打开文件进行读取
        text = file.read() # 读取文件的内容as str
        array=text.split('\n')
        for i in range(len(array)):
            array[i]=list(array[i])
        # print(array)
        #I set in simple_env_mine let map to be 1000(h)*800(w), I let the '1' in map is a round with radius=4,
        # so the map is 125*100 in txt
        p_n =np.sum([list.count(array[i],'p') for i in range(len(array))])
        c_n =np.sum([list.count(array[i],'c') for i in range(len(array))])
        num_obstacles=np.sum([list.count(array[i],'1') for i in range(len(array))])#np.sum() 有个特性，当数组为[True,False]时会累加True的个数
        # num_obstacles=0
        #所以当我们需要计算数组array中值value的个数时，使用语句：np.sum(data == value) 即可
        # print(num_obstacles)
        map=np.array(array,dtype='object')
        return map,p_n,c_n,num_obstacles
    
