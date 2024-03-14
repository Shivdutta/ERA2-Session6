import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary

class Model10(nn.Module):
    def __init__(self,dropout_value):
        super(Model10, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)
        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)

def get_summary(model, input_size) :   
    """
    Generate Model Summary Using torchsummary.

    This function provides a summary of the PyTorch model using the torchsummary library.
    It displays information such as the model architecture, number of parameters,
    and memory consumption.

    Parameters:
    - model (torch.nn.Module): PyTorch model for which the summary is to be generated.
    - input_size (tuple): Tuple representing the input size of the model, e.g., (channels, height, width).
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    return summary(network, input_size=input_size)


