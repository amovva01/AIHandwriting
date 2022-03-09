from data import IAMDataset
from model import HTR
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
from fastai.vision.all import show_image

def get_output(image, model):
    output = model(image)
    x = torch.argmax(output, dim=1)
    x = x.tolist()
    out = ""
    for c in x:
        out = out + chr(c-1)
    return out



model = torch.load("saves/model.pth")

model.cuda()


dataset = IAMDataset()
dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

for item in dataloader:
    target = item["label"]
    inputs = item["image"].cuda()

    output = get_output(inputs, model)

    targets = ""

    for i in target:
        targets = targets + chr(i.item()-1)

    print("Target:", targets)
    print("Output:", output)
    break

