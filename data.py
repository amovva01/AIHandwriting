from __future__ import print_function, division
from cProfile import label
import os
from unicodedata import normalize
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
        # converter = transforms.Compose([ transforms.ToPILImage(),
        #                                 transforms.Resize(224),
        #                                 transforms.ToTensor()])
        # normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = io.imread("data/images/" + self.images[idx])
        label = []
        for c in split(self.labels[idx]):
            label.append(ord(c)+1)
        # image = converter(image)[None, ...]
        # image = image.squeeze(0)
        # a = torch.zeros((3, image.size(1), image.size(2)))
        # a[0] = image
        # a[1] = image
        # a[2] = image

        # image = normalize(a)

    
        item = {'label': label, 'image': image}
        return item
        


        
