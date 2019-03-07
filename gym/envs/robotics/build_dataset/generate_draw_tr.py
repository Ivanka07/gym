import gym
import numpy as np


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []

def main():
    env = gym.make('FetchDrawTriangle-v1')
    numItr = 100
    initStateSpace = "random"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print('old obs', obs)
        print("ITERATION NUMBER ", len(actions))
        draw_triangle(env, obs)


    fileName = "data_fetch"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"

    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file

def draw_triangle(env, lastObs):

    desired_goals  = np.array(lastObs['desired_goal'])
    desired_goals = desired_goals.reshape((9,3))
    achieved_goals = np.array(lastObs['achieved_goal'])
    achieved_goals = achieved_goals.reshape((9,3))
    cur_grip_pos = lastObs['observation'][0:3]
    
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    i = 0
    g = 0
    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)
    
    while True:
        print('i=',i)
        env.render()
        
        if i < 9:
            g = i
        else:
            g = 0

        a = desired_goals[g,:] - cur_grip_pos
        act = [a[0], a[1], a[2], 0.05]        
        
        newObs, reward, done, info = env.step(act)
       #print('New obs' , newObs)
        achieved_goals = np.array(newObs['achieved_goal'])
        achieved_goals = achieved_goals.reshape((9,3))
        cur_grip_pos = newObs['observation'][0:3]
        print('Reward=', reward)
        print('Info=', info)
        done = info['is_success']
        if done: 
            print('Episode length = ', timeStep)
             
        if np.sum(achieved_goals[g,:]) > 0.0 and not done:
            i = i+1

        episodeAcs.append(act)
        episodeInfo.append(info)
        episodeObs.append(newObs)

        timeStep +=1    

        if timeStep >= env._max_episode_steps: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)


if __name__ == "__main__":
    main()
