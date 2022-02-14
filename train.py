from statistics import mode
from model import CNN
from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from fastai.vision.all import show_image



images = GeneratorFromRandom(
    length=10,
    count=60000,
    random_blur=True,
    random_skew=True,
    is_handwritten=True
)

device = "cpu"

maxseqlen=100

model = CNN(maxlinelen=maxseqlen)

if device=="cuda":
    model.cuda()

optim = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0)

pbar = tqdm(images)

i=1
running_loss=0
for img, lbl in pbar:
    optim.zero_grad()

    target = torch.LongTensor([ord(char)+1 for char in lbl] + [0 for i in range(len([ord(char)+1 for char in lbl]), maxseqlen)]).to(device)
    converter = transforms.ToTensor().to(device)

    input = converter(img)[None, ...]
    input = torch.permute(input, (0, 2, 1, 3)).to(device)

    output = model(input)

    loss = criterion(output, target)
    loss.backward()
    optim.step()
    
    running_loss += loss.item()

    pbar.set_description("loss:{:.4f}".format(running_loss/(i)))

    
    i+=1