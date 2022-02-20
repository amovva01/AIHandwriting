import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, maxlinelen) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(6, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 128, kernel_size=5),
            nn.ReLU(),
            # nn.MaxPool2d(3),
            nn.Conv2d(128, 256, kernel_size=3),      
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(10),

            nn.Flatten(),
            nn.Linear(25600, 128),
            
        )
        self.encode = nn.LSTM(128, 257, bidirectional=True)
        self.decode = nn.LSTM(257, 257, bidirectional=True)
        self.linear = nn.Linear(514, 257)
        
        self.maxlinelen=maxlinelen
    def forward(self, src):
        x = self.cnn(src)
        x = x[None, :]

        # x = torch.permute(x, (2, 0, 1))
        
        x,hidden = self.encode(x)
        x = self.linear(x)

        x,hidden = self.decode(x, hidden)
        x = self.linear(x)

        out = torch.zeros((self.maxlinelen, 257))
        out[0] = x

        for i in range(1,self.maxlinelen):
            x,hidden = self.decode(x,hidden)
            x = self.linear(x)

            out[i]=x

        return out
