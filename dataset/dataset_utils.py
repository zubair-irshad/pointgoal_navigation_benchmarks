from torch.utils.data import Dataset
# from numpy import genfromtxt
import os
# import pandas as pd
import numpy as np
# from PIL import Image
import torch
# from utils import feature_engineering_poses
from expert import Expert

# class Dataset_RNN_1(Dataset):
#     def __init__(self, data_path, scene_dir, mode, config_path,num_scenes, num_episodes_per_scene, transform=None):

#         self.data_path   = data_path
#         self.mode        = mode
#         self.config_path = config_path
#         self.transform   = transform
#         self.scene_dir   = scene_dir
#         self.num_scenes = num_scenes
#         self.num_episodes_per_scene = num_episodes_per_scene

#     def read_images_and_actions(self):
#         expert   = Expert(self.data_path, self.scene_dir, self.mode, self.config_path, self.transform)
#         ims, acs = expert.read_observations_and_actions(self.num_scenes, self.num_episodes_per_scene)
#         return ims, acs

#     def __getitem__(self, index):
#         "Generates one sample of data"
#         # Select sample

#         images, actions = self.read_images_and_actions()

#         X = images[index]
#         y = actions[index]
#         return X, y

#     def __len__(self):
#         "Denotes the total number of samples"
#         count = self.num_scenes*self.num_episodes_per_scene
#         return count

class Dataset_RNN(Dataset):
    def __init__(self, images, actions):

        self.images  = images
        self.actions = actions

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.images[index]
        y = self.actions[index]
        return X, y

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.images)
