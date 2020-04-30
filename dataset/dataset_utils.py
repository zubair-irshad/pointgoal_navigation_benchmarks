from torch.utils.data import Dataset
import os
import numpy as np
import torch
from expert import Expert

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
