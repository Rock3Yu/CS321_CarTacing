from array import array
import sys
import random
import pygame
import numpy as np
# from model.PettingZoo.pettingzoo.mpe.simple.simple import Scenario 
sys.path.append("D:\\Source\\Project_Big\\model\\PettingZoo")
##absolute import, python will find the top package in sys.path, we show subpackage by '.'
from pettingzoo_pc.mpe.police_criminal import police_criminal as pc
import pettingzoo_pc.mpe._mpe_utils.policy  as policy_p
import pettingzoo_pc.mpe._mpe_utils.policy_cri  as policy_c



def draw_map(filename): #大小固定100*125
    file=open(filename,'w')
    array=np.full((100,125),'-')
    array[40:60,45:90]='1'

    array[20,20]='p'
    array[80,80]='c'
    array[20,100]='e'
    array[99,124]='p'
    # array=''.join(array)
    str_a=''
    for row in array:
        for i in row:
            str_a=str_a+(str(i))
        str_a=str_a+('\n')
    print(str_a)
    file.write(str_a)


if __name__=='__main__':
    first=pc.exchange()
    # filename = './maps/nobound_withland.txt'
    # draw_map(filename)
    policy_p=policy_p.TestPolicy()
    policy_c=policy_c.TestPolicy()

    env=pc.raw_env()

    # first.generator(filename,100)
    env.reset()
        # print( [entity.state.p_pos for entity in env.world.entities]) 
    cnt=0
    x,y=0,0
    for agent in env.agent_iter(1200):
        env.render()
        

        # env.last()

        reward=0
        if agent[0:3]!='adv':
            agent_Id=int(agent[-1])+1
            real_agent=env.world.agents[agent_Id]
            observation  = env.scenario.observation(real_agent,env.world)
            reward = env.scenario.reward(real_agent,env.world)
            action=policy_p.action(observation, reward)  # type: ignore
        else :
            reward= env.scenario.adversary_reward(env.world.agents[0],env.world)
            observation  = env.scenario.observation(env.world.agents[0],env.world)
            action=policy_c.action(observation, reward)
        
        # print(observation)
        # if cnt==0 or cnt%10==0 or  cnt==599:
        #     input()
        # if cnt%30 ==0:
        #     x,y=random.random()*2-1,random.random()*2-1
        # action=np.array([0.,max(x,0.),max(-x,0.),max(y,0.),max(-y,0.)])
        env.step(action)
        cnt+=1
    # pygame.quit()
    env.close()

    