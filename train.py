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



images = GeneratorFromRandom(
    count=1000,
    random_blur=True,
    random_skew=True,
    is_handwritten=True
)

maxseqlen=100

model = CNN(maxlinelen=maxseqlen)

optim = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0)

for img, lbl in images:
    optim.zero_grad()

    target = torch.LongTensor([ord(char)+1 for char in lbl] + [0 for i in range(len([ord(char)+1 for char in lbl]), maxseqlen)])
    converter = transforms.ToTensor()

    input = converter(img)[None, ...]
    input = torch.permute(input, (0, 2, 1, 3))

    output = model(input)

    loss = criterion(output, target)
    loss.backward()
    optim.step()
    