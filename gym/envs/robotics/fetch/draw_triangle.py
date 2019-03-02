import os
import gym
from gym import utils
from gym.envs.robotics import fetch_env, rotations, robot_env
import numpy as np
import copy
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

        self.initial_goals = {
                    'goal0': [1.20727302, 0.5951655 , 0.49858342], 
                    'goal1': [1.22474433, 0.67346458, 0.64335482], 
                    'goal2': [1.22474433, 0.6348449 , 0.56827542], 
                    'goal3': [1.22474433, 0.723543  , 0.71355354], 
                    'goal4': [1.22474434, 0.76215388, 0.6341726 ], 
                    'goal5': [1.22474433, 0.80084932, 0.56561652], 
                    'goal6': [1.22474433, 0.85758673, 0.5013173 ], 
                    'goal7': [1.22474433, 0.76535415, 0.50012584], 
                    'goal8': [1.22474433, 0.68332691, 0.49927819]
                    }

        self.num_goals = num_goals
        self.goals = []
        self.ep_reward = 0
        self.reach_factor = 1.0
        self.last_reached_goal = -1
        self.names_to_pos = {}


        print('MODEL_XML_PATH', MODEL_XML_PATH)
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, gripper_rot=[1., 0., 0., 0.])
        utils.EzPickle.__init__(self)
        self.names_to_pos = copy.deepcopy(self.init_goals()) 
        print('init self.names_to_pos', self.names_to_pos)
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
       # self._sample_goals()


  #  def _render_callback(self):
  #      # Visualize target.
  #      sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()       
  #      site_id = self.sim.model.site_name2id('target0')
  #      self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
  #      self.sim.forward()


    def _render_callback(self):
        #print('Rendering callback')
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        for i in range(self.num_goals):
            site_id = self.sim.model.site_name2id('target'+str(i))
            #print('sites_offset[i]', sites_offset[i] )
            #print('self.goals[i].position ', self.goals[i].position )
            self.sim.model.site_pos[site_id] =  self.goals[i].position - sites_offset[i] 
            #print('self.sim.model.site_pos[site_id]', self.sim.model.site_pos[site_id])
        self.sim.forward()


    def init_goals(self):
        body_names = self.sim.model.body_names
        body_pos = self.sim.model.body_pos
        #names_to_pos = ids_to_pos(body_names, body_pos)
        return ids_to_pos(body_names, body_pos)

    


    def _sample_goals(self):
        self.goals = []
       # print('Pos in model ', self.init_goals())
        for k,v in self.initial_goals.items():
            random = np.random.uniform(-0.005, 0.005, size=3)
            v[0] =  v[0] #0.7 * (v[0]  + random[0]) + 0.6
            v[1] =  v[1] + random[1] #0.7 * (v[1]  + random[1]) + 0.4
            v[2] =  v[2] + random[2] #0.7 * (v[2] * 0.8  + random[2]) + 0.3
            g_id = int(k.split('goal')[1])     
            goal_obj = Goal(g_id, v, False)
            self.goals.append(goal_obj)
        return self.goals



    def _get_obs(self):
       # self._sample_goals()
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
                #print('Goal with id=',g.id, ' has a position=', goal)
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




    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        self._sample_goals()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        obs = self._get_obs()
        self.ep_reward = 0
        self.last_reached_goal = -1
        return obs

        
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        # todo: penalize if the arm is to far from the goal 
        #store in self.last_reached_goal id of the last reached goal
        reward = self.ep_reward
       # print('Computing reward for drawing trinagle')
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        cur_grip_position = self.sim.data.get_site_xpos('robot0:grip')
        
        for i in range(len(self.goals)):
            goal = self.goals[i]
            if not goal.reached and goal.id == (self.last_reached_goal + 1):
        #        print('Considering goal = ', goal.id, ' is goal reached? = ', goal.reached)
                dist_to_goal = distance_goal(grip_pos, goal.position)
         #       print('Dist to goal=', dist_to_goal)
                if dist_to_goal < self.distance_threshold:
          #          print('Reached goal with id =', goal.id)
                    self.goals[i].reached = True
                    self.last_reached_goal = goal.id
                    self.ep_reward +=1
                else:
                    reward = -1 * dist_to_goal
                break


        reward = self.ep_reward + reward
        #print('Reward = ', reward)
        #print('Episode Reward = ', self.ep_reward)
        return reward
 