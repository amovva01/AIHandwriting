from __future__ import print_function, division
from cProfile import label
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def split(word):
    return [char for char in word]

class IAMDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
        lineasciifile = open("data/lines.txt")
        self.labels = []
        self.images = []
        for line in lineasciifile:
            line = line.split(" ")

            if line[0] != "#":
                words = line[8].replace(" ", "")
                words = words.replace("|", " ")
                filename = line[0] + ".png"
                self.labels.append(words)
                self.images.append(filename)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        converter = transforms.ToTensor()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread("data/images/" + self.images[idx])
        label = []
        for c in split(self.labels[idx]):
            label.append(ord(c)+1)
        image = converter(image)[None, ...]
        image = image.squeeze(0)

        item = {'label': label, 'image': image}
        return item
        


        
