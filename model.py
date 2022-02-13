import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, maxlinelen) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.AdaptiveMaxPool2d(32),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(10),

            nn.Flatten(),
            nn.Linear(51200, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.encoderlstm = nn.LSTM(128, 257, num_layers=4)
        self.decoderlstm = nn.LSTM(257, 257, num_layers=4)
        self.maxlinelen=maxlinelen
    def forward(self, src):
        x = self.cnn(src)[None, ...]

        # x = torch.permute(x, (2, 0, 1))
        print(x.size())

        
        _,hidden = self.encoderlstm(x)

        x,hidden = self.decoderlstm(0, hidden)
        out = torch.zeros((self.maxlinelen, 257))
        out[0] = x

        for i in range(1,self.maxlinelen):
            x,hidden = self.decoderlstm(x,hidden)
            out[i]=x

        return 
