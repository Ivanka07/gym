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

    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
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
	trajectory_np = np.array(obj_trajectory)
	print('Trajectory shape', trajectory_np.shape)
	return np.mean(trajectory_np)


def transform_point():
	pass


def main(args):
	hand_tr_list, obj_tr_list = load_tr_as_np(args.human_trajectories, args.field1, args.field2)
	for tr in hand_tr_list:
		print(len(tr))

	print('hand_tr_list shape', hand_tr_list.shape)
	print('obj_tr_list', obj_tr_list.shape)


if __name__ == '__main__':
    args = argparser()
    main(args)
