#!/usr/bin/python3
import argparse
import gym
import math
import logging
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

"""
We need (obs,act) pairs in order to shuffle them to stabilize training
"""

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human_trajectories', default='reach_expert_trajectories.npz')
    parser.add_argument('--field1', default='hand_trajectories')
    parser.add_argument('--field2', default='obj_trajectories')
    parser.add_argument('--empty_world_model', default='empty_world.xml')
    parser.add_argument('--i', default=0)
    
    parser.add_argument('--gamma', default=0.998)
    parser.add_argument('--iterations', default=int(1e4), type=int)
    parser.add_argument('--env', default='FetchDrawTriangle-v1')
    parser.add_argument('--expert_file', default='data_fetch_reach_random_200.npz')
    parser.add_argument('--acs', default='actions.csv')
    parser.add_argument('--log_actions', default='log_actions')
    parser.add_argument('--max_episode_length', default=50, type=int)
    parser.add_argument('--render', default=1, choices=[0,1], type=int)
    parser.add_argument('--success_num', default=15, type=int)
    parser.add_argument('--num_timesteps', default=5000, type=int)
    parser.add_argument('--demo_file', default='data_fetch_reach_random_50.npz')
    parser.add_argument('--network', default='mlp')
    parser.add_argument('--seed', default=int(1), type=int)
    parser.add_argument('--num_env', default=int(1), type=int)
    parser.add_argument('--batch_size', default=int(4000), type=int)
    parser.add_argument('--policy_file', default='/policies/gail_her/gail1000')
    #parser.add_argument('--npz_file_name', default='')
    return parser.parse_args()


def load_tr_as_np(path, field1, field2):
	print('Loading from ', path)
	data = np.load(path)
	print('data keys', data.keys())
	return data[field1], data[field2]


def get_object(obj_trajectory):
	"""
	only for static objects and FetchReach env
	build the avarage obj
	"""
	trajectory_np = np.array(obj_trajectory, dtype=np.float32)
	print('Trajectory shape', trajectory_np.shape)
	return np.mean(trajectory_np, axis=0)


def add_goal_to_env_xml(input_xml, goal, output_xml, center_of_mass=[]):

    print('Input xml is=', input_xml)
    print('Goal', goal)
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
    tree.write( output_xml + '.xml')



def add_goals_to_env_xml(input_xml, goals, output_xml, center_of_mass=[]):
    if not len(goals):
        warnings.warn('List of goals is empty')

    if 'csv' in output_xml:
        output_xml = output_xml.split('csv')[0]

    print('Input xml is=', input_xml)
    print('Goals', goals[1:2])
    tree = ET.parse(input_xml)
    root = tree.getroot()
    world = root.find('worldbody')


    for i in range(len(goals)):
        if i > config['FetchDrawTriangle-v1']['limit_goals']:
            break

        pos = goals[i]
        print('Goal=', pos)
        body_attr ={
            'name': 'goal:g' + str(i),
            'pos': '{} {} {}'.format(pos[0], pos[1], pos[2])
        }

        site_attr ={
            'name': 'target0:id' + str(i),
            'pos': '{} {} {}'.format(0.0, 0.0, 0.0),
            'size': '{} {} {}'.format(0.02, 0.0, 0.02),
            'rgba':'{} {} {} {}'.format(1, 0, 0, 1), 
            'type': 'sphere'
        }

        body = Element('body', attrib=body_attr)
        site = SubElement(body, 'site', attrib=site_attr)
        world.append(body)
    tree.write( output_xml + '.xml')


def transform_point(object_pos):
	pass


def main(args):
    num = int(args.i)
    hand_tr_list, obj_tr_list = load_tr_as_np(args.human_trajectories, args.field1, args.field2)
	#for tr in hand_tr_list:
	#	print(len(tr))
    print('hand_tr_list shape', hand_tr_list.shape)
    obj_pos = get_object(obj_tr_list[num,])
    print('Obj_pos =', obj_pos)
   
    #print('Obj_pos =', str(obj_pos[0]-0.25),  str(obj_pos[1]+ 0.8441) , str(obj_pos[2]))
    output_worl = 'FetchReach_' + str(obj_pos[0]-0.3) + '_' + str(obj_pos[1]+ 0.8441) + '_' + str(obj_pos[2])
    add_goal_to_env_xml(args.empty_world_model, obj_pos, 'worlds/' + output_worl)


if __name__ == '__main__':
    args = argparser()
    main(args)
