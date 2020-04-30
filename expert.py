import os
import shutil

import numpy as np

import habitat
import torch
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower,action_to_one_hot
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from PIL import Image

cv2 = try_cv2_import()

class SimpleRLEnv(habitat.RLEnv):
	def get_reward_range(self):
		return [-1, 1]

	def get_reward(self, observations):
		return 0

	def get_done(self, observations):
		return self.habitat_env.episode_over

	def get_info(self, observations):
		return self.habitat_env.get_metrics()


class Expert:
	def __init__(self, data_path, scene_dir, mode, config_path,transform=None):
		self.data_path 	 = data_path
		self.mode 	     = mode
		self.config_path = config_path
		self.transform   = transform
		self.scene_dir   = scene_dir

	def read_observations_and_actions(self, num_trajectories, min_dist, max_dist):
		config                    = habitat.get_config(config_paths = self.config_path)
		config.defrost()
		config.DATASET.DATA_PATH  = self.data_path
		config.DATASET.SCENES_DIR = self.scene_dir
		# config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
		config.TASK.SENSORS.append("HEADING_SENSOR")
		config.freeze()

		env = SimpleRLEnv(config=config)
		goal_radius = env.episodes[0].goals[0].radius

		if goal_radius is None:
			goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
		follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)
		follower.mode = self.mode
		print("Environment creation successful")

		im_out = []
		ac_out = []
		for episode in range(num_trajectories):

			images =[]
			actions=[]	
			observations= env.reset()
			# Get first image 
			im = observations["rgb"]
			im = Image.fromarray(im)
			if self.transform is not None:
				im = self.transform(im) 
			images.append(im)

			geodesic_distance =  env.habitat_env.current_episode.info['geodesic_distance']
			if geodesic_distance >max_dist or geodesic_distance<min_dist:
				continue      
			print("Agent stepping around inside environment.")
		
			while not env.habitat_env.episode_over:
				best_action = follower.get_next_action(
					env.habitat_env.current_episode.goals[0].position
				)
				# one_hot = action_to_one_hot(int(0 if best_action is None else best_action))
				# act = torch.tensor(int(0 if best_action is None else best_action))
				actions.append(torch.tensor(int(0 if best_action is None else best_action)))
				
				if best_action is None:
					break
				observations, reward, done, info = env.step(best_action)
				im = observations["rgb"]
				im = Image.fromarray(im)
				if self.transform is not None:
					im = self.transform(im) 
				#Append images and actions for one trajectory	
				images.append(im)

			# actions = torch.LongTensor(actions).unsqueeze(0)
			# actions = torch.tensor(actions)
			images = torch.stack(images, dim=0)
			actions = torch.stack(actions, dim=0)

			# actions = torch.stack(actions, dim=0)

			# Some episodes run for 500 timesteps and are usually not optimal. Not including those episodes as part of the data here:
			if(images.shape[0])>150:
				continue

			im_out.append(images)
			ac_out.append(actions)
			print("Episode finished")

		return im_out, ac_out

