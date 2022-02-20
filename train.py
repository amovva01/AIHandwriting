from data import IAMDataset
from model import CNN
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
from fastai.vision.all import show_image


print("Loading dataset from data folder")
dataset = IAMDataset()
dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
print("Loaded data!")

device = "cuda"

maxseqlen=100

model = CNN(maxlinelen=maxseqlen)

if os.path.exists("saves/model.pth"):
    model = torch.load("saves/model.pth")

model.cuda()

print(next(model.parameters()).device)

optim = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

pbar = tqdm(dataloader)

for epoch in range(1, 10):
    i=1
    running_loss=0
    for item in pbar:

        target = item["label"]
        input = item["image"]

        optim.zero_grad()

        a = torch.zeros(maxseqlen)
        target = torch.LongTensor(target)
        a[:target.size(0)] = target
        target = a.type(torch.LongTensor).to(device)

        input = input.to(device)

        output = model(input)

        output=output.to("cuda")
        loss = criterion(output, target)
        loss.backward()
        optim.step()
        
        running_loss += loss.item()

        pbar.set_description("loss:{:.4f}".format(running_loss/(i)))

        i+=1

        torch.cuda.empty_cache()
    torch.save(model, "saves/model.pth")