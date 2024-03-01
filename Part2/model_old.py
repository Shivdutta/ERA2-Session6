import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary

# r
# class Net(nn.Module):
#     def __init__(self,dropout_prob):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 4, 3, padding=1, bias=False),
#             #nn.ReLU(), #need to check
#             nn.BatchNorm2d(4),
#             nn.Conv2d(4, 4, 3, padding=1, bias=False),
#             nn.BatchNorm2d(4),
#             nn.Conv2d(4, 4, 3, padding=1, bias=False),
#             nn.BatchNorm2d(4)
#         )
#         self.trans1 = nn.Sequential(
#             nn.MaxPool2d(2,2),
#             nn.Dropout(p=dropout_prob)
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 8, 3, padding=1, bias=False),
#             #nn.ReLU(), #need to check
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 8, 3, padding=1, bias=False),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 8, 3, padding=1, bias=False),
#             nn.BatchNorm2d(8)
#         )
#         self.trans2 = nn.Sequential(
#             nn.MaxPool2d(2,2),
#             nn.Dropout(p=dropout_prob)
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(8, 8, 3, padding=1, bias=False),
#             #nn.ReLU(), #need to check
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 8, 3, padding=1, bias=False),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 8, 3, padding=1, bias=False),
#             nn.BatchNorm2d(8)
#         )
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.flat = nn.Conv2d(8,10,1)


#     def forward(self, x,droupout=True):
#         x = self.conv1(x)
#         x = self.trans1(x)
#         x = self.conv2(x)
#         x = self.trans2(x)
#         x = self.conv3(x)
#         x = self.gap(x)
#         x = self.flat(x)
#         x = x.view(-1,10)
#         x = F.log_softmax(x ,dim=1)
#         return x

# m
# class Net(nn.Module):
#     def __init__(self,dropout_prob):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 8, 3, padding=1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(8),
#             nn.Dropout2d(p=dropout_prob)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(8, 16, 3, padding=1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout2d(p=dropout_prob)
#         )

#         self.pool1= nn.MaxPool2d(2, 2)

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(16, 24, 3, padding=1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(24),
#             nn.Dropout2d(p=dropout_prob)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(24, 16, 3, padding=1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout2d(p=dropout_prob)
#         )

#         self.pool2= nn.MaxPool2d(2, 2)

#         self.conv5 = nn.Sequential(
#             nn.Conv2d(16, 16, 3, padding=0, bias=False),
#             nn.Conv2d(16, 14, 1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(14),
#             nn.Dropout2d(p=dropout_prob)
#         )


       
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(14, 14, 3, padding=0, bias=False),
#         )

#         self.global_avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(14, 10)  
      
    
#     def forward(self, x):
#         x = self.conv1(x)     
#         x = self.conv2(x)
#         x = self.pool1(x)
#         x = self.conv3(x)
#         x = self.conv4(x)

#         x = self.pool2(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
               
#         x = self.global_avgpool(x)
#         x = x.view(-1, 14)
#         x = self.fc(x)
#         return F.log_softmax(x)

# s
# class Net(nn.Module):
#     def __init__(self,dropout_prob):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 4, 3, padding=1, bias=False),
#             nn.ReLU(), #need to check
#             nn.BatchNorm2d(4),
#             nn.Conv2d(4, 4, 3, padding=1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(4),
#             nn.Conv2d(4, 4, 3, padding=1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(4),
#             nn.Dropout2d(p=dropout_prob)            
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 8, 3, padding=1, bias=False),
#             nn.ReLU(), #need to check
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 8, 3, padding=1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 8, 3, padding=1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(8),
#             nn.Dropout2d(p=dropout_prob)
#         )
        
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(8, 16, 3, padding=1, bias=False),
#             nn.ReLU(), #need to check
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 16, 3, padding=1, bias=False),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 16, 3, padding=1, bias=False),
#             # nn.ReLU(),
#             # nn.BatchNorm2d(16),
#             # nn.Dropout2d(p=dropout_prob)
#         )

#         self.pool1= nn.MaxPool2d(2, 2)

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.flat = nn.Conv2d(16,10,1)


#     def forward(self, x,droupout=True):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.pool1(x)
#         x = self.gap(x)
#         x = self.flat(x)
#         x = x.view(-1,10)
#         x = F.log_softmax(x ,dim=1)
#         return x


# a
# class Net(nn.Module):
#     def __init__(self, drop: float = 0.0):
#         super().__init__()

#         # drop = 0.02 # droput value (drop 10% neurons)
#         self.input_layer = nn.Sequential(
#             nn.Conv2d(1, 4, 3, padding=1, bias=False),  # input: 28x28x1 output: 28x28x4 RF:3x3
#             nn.ReLU(),  # activation function relu
#             nn.BatchNorm2d(4),  # Batch normalization
#             nn.Dropout2d(drop),
#         )
#         # padding=1
#         # Block 1
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(4, 8, 3, bias=False),  # input: 28x28x4 output: 26x26x8 RF:5x5
#             nn.ReLU(),
#             nn.BatchNorm2d(8),
#             nn.Dropout2d(drop),
#             # nn.Conv2d(8, 8, 3, bias=False), # input: 28x28x16 output: 28x28x24 RF:7x7
#             # nn.ReLU(),
#             # nn.BatchNorm2d(8),
#             # nn.Dropout2d(drop),
#         )

#         # Transition Block 1x1
#         self.trans1 = nn.Sequential(
#             nn.MaxPool2d(2, 2),  # input: 26x26x8 output: 13x13x8 RF:14x14
#             nn.Conv2d(8, 4, 1, bias=False),  # input: 13x13x8 output: 13x13x4 RF:14x14
#             nn.ReLU(),
#             nn.BatchNorm2d(4),
#             nn.Dropout2d(drop),
#         )
#         # Block 2
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 16, 3, bias=False),  # input: 13x13x4 output: 11x11x16 RF:16x16
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout2d(drop),
#             nn.Conv2d(16, 32, 3, bias=False),  # input: 11x11x16 output: 9x9x32 RF:18x18
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Dropout2d(drop),
#         )

#         # Transition Block (1x1)
#         self.trans2 = nn.Sequential(
#             nn.Conv2d(32, 8, 1, bias=False),  # input: 9x9x32 output: 9x9x8 RF:18x18
#             nn.ReLU(),
#             nn.BatchNorm2d(8),
#             nn.Dropout2d(drop),
#         )

#         # Block 3
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(8, 16, 3, bias=False),  # input: 9x9x8, output: 7x7x16 RF: 20x20
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout2d(drop),
#             nn.Conv2d(16, 32, 3, bias=False),  # # input: 7x7x16 output: 5x5x32 RF:22x22
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Dropout2d(drop),
#         )

#         # GAP Layer
#         self.gap = nn.Sequential(nn.AvgPool2d(5))  # Global average pooling
#         # Fully Connected Layer
#         # Fully connected layer
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=10,
#                 kernel_size=(1, 1),
#                 padding=0,
#                 bias=False,
#             ),  # output  RF: 28
#         )

#     def forward(self, x):
#         x = self.input_layer(x)  # input in conv1 block
#         x = self.conv1(x)  # input in conv1 block
#         x = self.trans1(x)  # input in trnasition block 1
#         x = self.conv2(x)  # input in conv2 block
#         x = self.trans2(x)  # input in transition block 2
#         x = self.conv3(x)  # input in conv3 block
#         x = self.gap(x)  # global average pooling
#         x = self.conv4(x)

#         x = x.view(-1, 10)  # reshape 2d tensor to 1d
#         return F.log_softmax(x, dim=1)  # final prediction

# new sp
class Net(nn.Module):
    def __init__(self,dropout_prob):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            #nn.SiLU(), 
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            #nn.SiLU(), 
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            #nn.SiLU(), 
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=dropout_prob)            
        )
        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            #nn.SiLU(), 
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            #nn.SiLU(), 
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            #nn.SiLU(), 
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=dropout_prob)
        )
        self.trans2 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            #nn.SiLU(), 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 24, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout2d(p=dropout_prob)
        )
        
        self.trans3 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )
        self.conv4 = nn.Sequential(
             nn.Conv2d(24, 16, 3, padding=1, bias=False),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Conv2d(16, 14, 3, padding=1, bias=False),
            #  nn.ReLU(),
            #  nn.BatchNorm2d(14),
            #  nn.Conv2d(14, 14, 3, padding=1, bias=False)
             )
        
        self.pool1= nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(14, 10) 

    def forward(self, x,droupout=True):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.conv3(x)
        x = self.trans3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.flat(x)
        #x = x.view(-1,10)
        x = self.fc(x)  # Linear layer
        x =  F.log_softmax(x ,dim=1)
        return x

#s p
# class Net(nn.Module):
#     def __init__(self,dropout_prob):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 4, 3, padding=1, bias=False),
#             #nn.SiLU(), 
#             nn.ReLU(),
#             nn.BatchNorm2d(4),
#             nn.Conv2d(4, 4, 3, padding=1, bias=False),
#             #nn.SiLU(), 
#             nn.ReLU(),
#             nn.BatchNorm2d(4),
#             nn.Conv2d(4, 4, 3, padding=1, bias=False),
#             #nn.SiLU(), 
#             nn.ReLU(),
#             nn.BatchNorm2d(4),
#             nn.Dropout2d(p=dropout_prob)            
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 8, 3, padding=1, bias=False),
#             #nn.SiLU(), 
#             nn.ReLU(),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 8, 3, padding=1, bias=False),
#             #nn.SiLU(), 
#             nn.ReLU(),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, 8, 3, padding=1, bias=False),
#             #nn.SiLU(), 
#             nn.ReLU(),
#             nn.BatchNorm2d(8),
#             nn.Dropout2d(p=dropout_prob)
#         )
        
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(8, 16, 3, padding=1, bias=False),
#             #nn.SiLU(), 
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 16, 3, padding=1, bias=False)            
#         )

#         # self.conv4 = nn.Sequential(
#         #     nn.Conv2d(16, 32, 3, padding=0, bias=False),
#         #     # nn.ReLU(),
#         #     # nn.BatchNorm2d(14),
#         #     # nn.Conv2d(14, 14, 3, padding=0, bias=False)
#         # )

#         self.pool1= nn.MaxPool2d(2, 2)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         #self.flat = nn.Flatten()
#         #self.fc = nn.Linear(32, 10)  


#    def forward(self, x,droupout=True):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = self.conv3(x)
    #     #x = self.conv4(x)
    #     #x = self.pool1(x)
    #     #x = self.gap(x)
    #     #x = self.flat(x)
    #     #x = x.view(-1,10)
    #     #x = self.fc(x)  # Linear layer
    #     #x= = F.log_softmax(x ,dim=1)
    #     return x


def print_model_summary(dropout_prob, inputsize):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(dropout_prob=dropout_prob).to(device)
    summary(model, input_size=inputsize)