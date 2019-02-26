import gym
import numpy as np
from gym.spaces import Box, Discrete, Dict


print('Current gym version =', gym.__version__)
env = gym.make('FetchDrawTriangle-v1')
#env = gym.make('FetchPush-v1')

obs = env.reset()
print('Observation space= ', env.observation_space)
print('Action space=', env.action_space.shape)
"""{'observation': array([ 1.31642683e+00,  3.94268249e-01,  7.01529477e-01,  1.62232554e+00,
        6.92140471e-01,  2.48922446e-02,  3.05898702e-01,  2.97872223e-01,
       -6.76637232e-01,  5.03188288e-02,  5.03330159e-02, -1.57079633e+00,
        1.55544866e+00,  1.57079633e+00, -8.78151430e-03, -1.06652152e-02,
        6.51380250e-03,  2.55037841e-17, -1.62687594e-15, -3.37040479e-19,
        8.78151430e-03,  1.06652152e-02, -6.51380250e-03, -9.90582143e-03,
       -1.08097720e-02]), 
    'achieved_goal': array([1.62232554, 0.69214047, 0.02489224]), 
    'desired_goal': array([1.47274893, 0.65727962, 0.86690607])}
"""
obs_arr = np.empty([31])
is_dict = 'dict' in str(type(obs))
print('Is dict?', is_dict)

#for key, value in obs.items():
	#value = value.reshape((value.shape[0]))
	#np.concatenate((obs_arr, value))

#obs_arr =  obs_arr.reshape((obs_arr.shape[1], obs_arr.shape[0] ))
#print(obs_arr.reshape((obs_arr.shape[1],obs_arr.shape[0] )))

while True:
	env.render()
	
	a = env.action_space.sample()
	#print('Sampled a=', a)
	obs, reward, done, info = env.step(a)
	print(obs)
	#print('New observation=', obs)