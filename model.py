import torch
import torch.nn as nn

# class CNN(nn.Module):
#     def __init__(self, maxlinelen) -> None:
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 6, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(6, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 128, kernel_size=3),
#             nn.ReLU(),
#             # nn.MaxPool2d(3),
#             nn.Conv2d(128, 256, kernel_size=3),      
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(10),
#             nn.ReLU(),

#             nn.Flatten(),
#             nn.Linear(25600, 2048),
            
#         )
#         self.encode = nn.LSTM(2048, 257, bidirectional=True)
#         self.decode = nn.LSTM(257, 257, bidirectional=True)
#         self.linear = nn.Linear(514, 257)
        
#         self.maxlinelen=maxlinelen
#     def forward(self, src):
#         x = self.cnn(src)
#         x = x[None, :]

#         # x = torch.permute(x, (2, 0, 1))
        
#         x,hidden = self.encode(x)
#         x = self.linear(x)

#         x,hidden = self.decode(x, hidden)
#         x = self.linear(x)

#         out = torch.zeros((self.maxlinelen, 257))
#         out[0] = x

#         for i in range(1,self.maxlinelen):
#             x,hidden = self.decode(x,hidden)
#             x = self.linear(x)

#             out[i]=x

#         return out

class HTR(nn.Module):
    def  __init__(self, hidden_dims, numheads, numencodelayers, numdecodelayers, maxlinelen) -> None:
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        del self.resnet.fc
        self.convolutional_layer = nn.Conv2d(2048, hidden_dims, 1)

        self.encode_layers = nn.TransformerEncoderLayer(hidden_dims, numheads)
        self.encoder = nn.TransformerEncoder(self.encode_layers, numencodelayers)

        self.decode_layers = nn.TransformerEncoderLayer(hidden_dims, numheads)
        self.decoder = nn.TransformerEncoder(self.decode_layers, numdecodelayers)

        self.fc = nn.Linear(256, 257)

        self.fc2 = nn.Linear(256*2*maxlinelen, 256)
        self.fc3 = nn.Linear(512, 256)

        self.encode = nn.LSTM(256, 256, bidirectional=True)
        self.decode = nn.LSTM(256, 256, bidirectional=True)
        
        self.pool = nn.AdaptiveMaxPool2d(10)

        self.maxlinelen=maxlinelen
    
    def throughresnet(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)   
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x

    def forward(self, input):
        x = self.throughresnet(input)

        x = self.pool(x)

        # x = self.encoder(x)
        # x = self.decoder(x)
        # x = self.fc(x)
        x = x.flatten(1,3)
        x=self.fc2(x)
        x = x[None, :]


        x,hidden = self.encode(x)
        x = self.fc3(x)

        x,hidden = self.decode(x, hidden)
        x = self.fc3(x)

        out = torch.zeros((self.maxlinelen, 257))
        b = self.fc(x)
        out[0] = b

        for i in range(1,self.maxlinelen):
            x,hidden = self.decode(x,hidden)
            x = self.fc3(x)

            out[i]=self.fc(x)

        return out