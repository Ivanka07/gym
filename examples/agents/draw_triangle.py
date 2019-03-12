import gym
import numpy as np
from gym.spaces import Box, Discrete, Dict


print('Current gym version =', gym.__version__)
env = gym.make('FetchDrawTriangle-v1')
obs = env.reset()
print('Observation space= ', env.observation_space)
print('Action space=', env.action_space.shape)
print('Max episode length ', env._max_episode_steps)

desired_goals  = np.array(obs['desired_goal'])
desired_goals = desired_goals.reshape((9,3))
achieved_goals = np.array(obs['achieved_goal'])
achieved_goals = achieved_goals.reshape((9,3))

print('desired_goals=', desired_goals)
i = 0
ep_length = 0 

goal = []
achiev_goal = []
dginx = 0

num_iterations = 1
for i in range(num_iterations):
    while True:
        i += 1
        
        if i == env._max_episode_steps:
            print('achieved maximal')
            obs = env.reset()
            dginx = 0
            i = 0

           # break
        
        env.render()
   
        cur_grip_pos = obs['observation'][0:3]
        #print(cur_grip_pos)
        a = (desired_goals[dginx,:] - cur_grip_pos) 
        print(a)
        act = [a[0], a[1], a[2], 0.15]
        obs, reward, done, info = env.step(act)
        
        achieved_goals = np.array(obs['achieved_goal'])
        achieved_goals = achieved_goals.reshape((9,3))
        desired_goals  = np.array(obs['desired_goal'])
        desired_goals = desired_goals.reshape((9,3))
        
        #print(achieved_goals)
        print(desired_goals[dginx,:])
        ag = achieved_goals[dginx,:]
        dg = desired_goals[dginx,:]

        dist = np.linalg.norm(np.array(ag) - np.array(dg), axis=-1)
        print('dist = ', dist)
        if (dist < 0.05) and dginx < 8 :
            dginx +=1

       
        #print('Reward = ', reward)
        #print('Info=', info)
        done = info['is_success']
    
        if done:
            print('Episode length = ', i)
            #break
    