#!/usr/bin/env python

import gym
import numpy as np
import csv
import argparse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import datetime
import pkg_resources
import cfg_load
import warnings
import glob
from config import goals_config

#better is if we read header from a config
def read_gt_data_csv(file):
	goals = []
	with open(file, newline='') as csv_data:
		reader = csv.DictReader(csv_data, delimiter=',')
		headers = reader.fieldnames
		assert(headers[1] ==  'field.x' and headers[2] ==  'field.y' and headers[3] ==  'field.z'), 'Check header in csv file. Expected: filed.[x|y|z]'
		for line in reader:
			pos = [float(line['field.x']),float(line['field.y']),float(line['field.z'])]
			goals.append(pos)
	return goals


def build_triangle_from_goals(goals):
	min_y = []
	max_y = []
	max_z = []

	min_y_dist = 100
	max_y_dist = 0
	max_z_dist = 0

	_goals = []

	for goal in goals:
		if goal[1] < min_y_dist:
			min_y_dist = goal[1]
			min_y = goal
		
		if goal[1] > max_y_dist:
			max_y_dist = goal[1]
			max_y = goal

		if goal[2] > max_z_dist:
			max_z_dist = goal[2]
			max_z = goal

	print('Found usefull goals', max_z, min_y, max_y)

	_goals.append(min_y)
	_goals.append(max_y)
	_goals.append(max_z)

	return _goals

def calc_center_of_mass(goals):
	center_of_mass = [0, 0, 0]
	for g in goals:
		center_of_mass[0] += g[0] 
		center_of_mass[1] += g[1]
		center_of_mass[2] += g[2]

	center_of_mass[0] = center_of_mass[0] / len(goals)
	center_of_mass[1] = center_of_mass[1] / len(goals)
	center_of_mass[2] = center_of_mass[2] / len(goals)

	print('calculated center of  center ', center_of_mass)
	return center_of_mass


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


