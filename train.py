from data import IAMDataset
from model import HTR
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

model = HTR(256, 8, 6, 6, maxlinelen=100)

if os.path.exists("saves/model.pth"):
    model = torch.load("saves/model.pth")

model.cuda()

print(next(model.parameters()).device)

optim = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

for epoch in range(1, 100):
    print("Next epoch starting:", epoch)
    pbar = tqdm(dataloader)

    i=1
    running_loss=0
    for item in pbar:

        try:
            target = item["label"]
            inputs = item["image"].cuda()


            a = torch.zeros(maxseqlen)
            target = torch.LongTensor(target)
            a[:target.size(0)] = target
            target = a.type(torch.LongTensor).to(device)

            output = model(inputs).cuda()
            loss = criterion(output, target)
            loss.backward()
            optim.step()

            optim.zero_grad()

            
            running_loss += loss.item()

            pbar.set_description("loss:{:.4f}".format(running_loss/(i)))

            i+=1
        except:
            print("There was an error. Ignoring.")

    print("Saving model:", epoch)
    torch.save(model, "saves/model.pth")
    print("Model saved")