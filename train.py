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

device = "cuda"

maxseqlen=100

model = CNN(maxlinelen=maxseqlen)


model.cuda()

print(next(model.parameters()).device)

optim = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

pbar = tqdm(images)

i=1
running_loss=0
for img, lbl in pbar:
    optim.zero_grad()

    target = torch.LongTensor([ord(char)+1 for char in lbl] + [0 for i in range(len([ord(char)+1 for char in lbl]), maxseqlen)]).to(device)
    converter = transforms.ToTensor()

    input = converter(img)[None, ...]
    input = torch.permute(input, (0, 1, 2, 3)).to(device)

    print(input.size())
    output = model(input)

    output=output.to("cuda")
    loss = criterion(output, target)
    loss.backward()
    optim.step()
    
    running_loss += loss.item()

    pbar.set_description("loss:{:.4f}".format(running_loss/(i)))

    
    i+=1