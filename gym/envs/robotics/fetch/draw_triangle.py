import os
import gym
from gym import utils
from gym.envs.robotics import fetch_env, rotations, robot_env
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'draw_triangle.xml')
fullpath = os.path.join(os.path.dirname(__file__), '../assets', MODEL_XML_PATH)

"""
x = nach vorne
+y = links in rob coord
+z = nach oben
"""

def distance_goal(goal_a, goal_b):
    if 'numpy.ndarray' != type(goal_a):
        goal_a = np.array(goal_a)

    if 'numpy.ndarray' != type(goal_b):
        goal_b = np.array(goal_b)

    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



def ids_to_pos(body_names, body_pos, goal_tag='goal'):
    assert len(body_names) == len(body_pos), 'Expected equal length of body_names and body_pos'
    goals_to_pos = {}
    for i in range(len(body_names)):
        name = body_names[i]
        pos = body_pos[i]
        if goal_tag in name:
            goals_to_pos[name] = pos           
    return goals_to_pos



class Goal():
    def __init__(self, id, position, reached=False):
        self.id = id
        self.position = position
        self.reached = reached

    def print(self):
        print('Position = {}, id = {}, reached = {} '.format(self.position, self.id, self.reached))


class FetchDrawTriangleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', num_goals=9):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }

        self.num_goals = num_goals
        self.goals = []
        self.ep_reward = 0
        self.reach_factor = 0.75
        self.last_reached_goal = -1

        print('MODEL_XML_PATH', MODEL_XML_PATH)
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.1,
            initial_qpos=initial_qpos, reward_type=reward_type, gripper_rot=[1., 0., 0., 0.])
        utils.EzPickle.__init__(self)
        #self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self._sample_goals()



    def _render_callback(self):
        print('Rendering callback')
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        for i in range(self.num_goals):
            site_id = self.sim.model.site_name2id('target'+str(i))
            self.sim.model.site_pos[site_id] =  self.goals[i].position - sites_offset[i] 
       #     print('self.sim.model.site_pos[site_id]', self.sim.model.site_pos[site_id])
        self.sim.forward()



    def _sample_goals(self):
        print('--- Sampling goals ---')
        body_names = self.sim.model.body_names
        body_pos = self.sim.model.body_pos
        names_to_pos = ids_to_pos(body_names, body_pos)
        self.goals = []
        for k,v in names_to_pos.items():
            random = np.random.uniform(-0.015, 0.015, size=3)
            v[0] =  self.initial_gripper_xpos[0] + v[0]/3.5 + random[0]
            v[1] =  self.initial_gripper_xpos[1] + v[1]/1.2 + random[1]
            v[2] =  self.initial_gripper_xpos[2] + v[2]/2.5 + random[2]      
            goal_obj = Goal(k, v, False)
            self.goals.append(goal_obj)
        return self.goals



    def _get_obs(self):
        desired_goals = []
        achieved_goals = []
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = gym.envs.robotics.utils.robot_get_obs(self.sim)

        #gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        obs = np.concatenate([grip_pos, grip_velp, gripper_vel])
        
        if len(self.goals) == 0:
            desired_goals = np.zeros((1, self.num_goals*3))
            achieved_goals = np.zeros((1,self.num_goals*3))
        else:
            for g in self.goals:
                goal = [g.position[0], g.position[1], g.position[2]]
                for g1 in goal:
                    desired_goals.append(g1)
                
                if g.reached:
                    for g_el in goal:
                        achieved_goals.append(g_el)
                else:
                    achieved_goals.append(0)
                    achieved_goals.append(0)
                    achieved_goals.append(0)

        return {
            'observation': obs.copy(),
            'achieved_goal': np.array(achieved_goals),
            'desired_goal': np.array(desired_goals),
        }


    def _is_success(self, achieved_goal, desired_goal):
        '''
        Hard condition: Success is True, when all of the goals are reached
        We will try soft condition: if 75% of all goals (scores) are reached -> done
        reward with additinal num_goals if all goals are reached
        '''
        has_to_be_reached = self.reach_factor * self.num_goals

        if has_to_be_reached <= self.ep_reward:
            print('Consider as done! Reward=', self.ep_reward)
        return has_to_be_reached <= self.ep_reward


    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        # todo: penalize if the arm is to far from the goal 
        #store in self.last_reached_goal id of the last reached goal
        reward = self.ep_reward
        print('Computing reward for drawing trinagle')
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        cur_grip_position = self.sim.data.get_site_xpos('robot0:grip')
        print('distance_threshold', self.distance_threshold)
        
        for i in range(len(self.goals)):
            goal = self.goals[i]
            print('Considering goal = ', goal.id, ' is goal reached? = ', goal.reached)
            if not goal.reached and goal.id == (self.last_reached_goal + 1):
                dist_to_goal = distance_goal(grip_pos, goal.position)
                print('Dist to goal=', dist_to_goal)
                if dist_to_goal < self.distance_threshold:
                    print('Reached goal with id =', goal.id)
                    self.goals[i].reached = True
                    self.last_reached_goal = goal.id
                    reward +=1
                    break
                else:
                    reward = -1 * dist_to_goal


        self.ep_reward = reward
        print('Reward = ', reward)
        return reward
 