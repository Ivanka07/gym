import gym
import numpy as np
from gym.spaces import Box, Discrete, Dict
from baselines import her
from pathlib import Path
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines import her
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

import os
import joblib




def add_goal_to_env_xml(goal, center_of_mass=[]):

    input_xml = "/home/ivanna/git/gym/gym/envs/robotics/build_dataset/empty_world.xml"
    tree = ET.parse(input_xml)
    root = tree.getroot()
    world = root.find('worldbody')

    body_attr ={
        'name': 'goal0:',
        'pos': '{} {} {}'.format(goal[0], goal[1], goal[2])
    }

    site_attr ={
        'name': 'target0',
        'pos': '{} {} {}'.format(0.0, 0.0, 0.0),
        'size': '{} {} {}'.format(0.02, 0.0, 0.02),
        'rgba':'{} {} {} {}'.format(1, 0, 0, 1), 
        'type': 'sphere'
    }

    body = Element('body', attrib=body_attr)
    site = SubElement(body, 'site', attrib=site_attr)
    world.append(body)
    output_xml = "/home/ivanna/git/gym/gym/envs/robotics/assets/fetch/FetchReach_1.5839665_-0.26136932_0.82435787"
    tree.write( output_xml + '.xml')







print('Current gym version =', gym.__version__)
env = gym.make('FetchReach-v1')
o = env.reset()
print('Observation space= ', env.observation_space)
print('Action space=', env.action_space.shape)

while True:
	env.render()
#policy = joblib.load('/home/ivanna/mt_data/experiments/gail/gail_her_model')

#data = np.load()
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
obs = []
acs = []


data= np.load('reach_expert_trajectories.npz')
actions = data['hand_trajectories']
"""_actions = [[float(x[0]),float(x[1]),float(x[2])] for x in actions[43,]]
a_len = len(_actions)
i=0
while True:
	env.render()
	if i >= (a_len-1):
		break
	wp = [_actions[i][0] , _actions[i][1]  , _actions[i][2]]
	#print('translated = ', wp)
	#a = [current_grip_pos[0,], current_grip_pos[1,], current_grip_pos[2,], 0.0] #wp - current_grip_pos
	a = wp - o['achieved_goal']
	a = [x for x in a[:]]
	a.append(0.0)
	
	if i > 100:
		obs.append(o)
		acs.append(a)

	#print('i = ', i)
	#a = env.action_space.sample()
	#print('Sampled a=', a)
	o, reward, done, info = env.step(a)
	i += 1
	if info['is_success']:
		print('i', i, ' is_success=', info, 'storing ',  num)
		np.savez(str(num) + '.npz', obs=obs, acs=acs)
		break
"""
g = None
for k in range(7, 82):
	#env.render()

	print('*********************************** new iteration = {} ******************************'.format(k))
	obs = []
	acs = []

	num = int(k)

	_actions = [[float(x[0]),float(x[1]),float(x[2])] for x in actions[num,]]
	#print(actions)
	a_len = len(_actions)
	print('Number of actions = ', a_len)	
	i=1
	while True:
		#print(o)

		#env.render()
		#a = policy.step(obs)
		grip = o['observation'][:4]
		current_grip_pos =  o['desired_goal'] - o['achieved_goal'] # obs['observation'][:4] #
		#print('untranslated = ', actions[i][0] , actions[i][1] , actions[i][2])
		wp = [_actions[i][0] , _actions[i][1]  , _actions[i][2]]
		#print('translated = ', wp)
		#a = [current_grip_pos[0,], current_grip_pos[1,], current_grip_pos[2,], 0.0] #wp - current_grip_pos
		a = wp - o['achieved_goal']
		#a.append(0.05)
		#print('action ', a)
		a = [x for x in a[:]]
		a.append(0.0)


		#print('i = ', i)
		#a = env.action_space.sample()
		#print('Sampled a=', a)
		o, reward, done, info = env.step(a)
		#print(a_len - 10)
		if i == (a_len - 100):
			g = wp
			print('saving new model with goal =', wp)
			add_goal_to_env_xml(wp)
			break
		else:
			#print('nope', i, 'a_len',  a_len)
			i+=1
	env = gym.make('FetchReach-v1')
	o = env.reset()
	i = 1

	while True:
		#env.render()
		print('Goal = ', g)
		print('goal in o', o['desired_goal'])

		if i >= (a_len-1):
			break
		wp = [_actions[i][0] , _actions[i][1]  , _actions[i][2]]
		#print('translated = ', wp)
		#a = [current_grip_pos[0,], current_grip_pos[1,], current_grip_pos[2,], 0.0] #wp - current_grip_pos
		a = wp - o['achieved_goal']
		a = [x for x in a[:]]
		a.append(0.0)
		
		
		obs.append(o)
		acs.append(a)

		#print('i = ', i)
		#a = env.action_space.sample()
		#print('Sampled a=', a)
		o, reward, done, info = env.step(a)
		i += 1
		if info['is_success']:
			print('i', i, ' is_success=', info, 'storing ',  num)
			np.savez(str(num) + '.npz', obs=obs, acs=acs)
			break
	