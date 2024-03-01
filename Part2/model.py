"""
Module Imports for PyTorch and Utility Libraries.

This module includes import statements for essential PyTorch modules and utilities
used in deep learning tasks. It incorporates modules for neural network construction
(nn), functional operations (F), tensor manipulation, training progress visualization
(tqdm), and model summary (torchsummary).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary

class Net(nn.Module):
    """
    A convolutional neural network (CNN) architecture implemented using PyTorch's nn.Module.

    Parameters:
    - dropout_prob (float): Probability of dropout for the Dropout2d layers in the network.

    Attributes:
    - conv1 (nn.Sequential): First convolutional block consisting of multiple convolutional layers,
      ReLU activation, batch normalization, and dropout.
    - trans1 (nn.Sequential): First max-pooling layer to downsample the spatial dimensions.
    - conv2 (nn.Sequential): Second convolutional block with similar structure to conv1.
    - trans2 (nn.Sequential): Second max-pooling layer.
    - conv3 (nn.Sequential): Third convolutional block with dropout applied.
    - trans3 (nn.Sequential): Third max-pooling layer.
    - conv4 (nn.Sequential): Fourth convolutional block without dropout.
    - pool1 (nn.MaxPool2d): Max-pooling layer after conv4.
    - gap (nn.AdaptiveAvgPool2d): Global average pooling layer to reduce spatial dimensions to 1x1.
    - flat (nn.Flatten): Flatten layer to convert the spatial dimensions to a flat vector.
    - fc (nn.Linear): Fully connected layer for classification with output size 10.

    Methods:
    - forward(x, dropout=True): Forward pass through the network.
    
    Example usage:
    ```python
    # Initialize the network with a specified dropout probability
    net = Net(dropout_prob=0.5)

    # Forward pass through the network with input tensor x
    output = net(x)
    ```

    Note:
    The network is designed for image classification tasks with a final output size of 10
    (assuming it is used for classifying into 10 classes) and includes dropout for regularization.

    """
    def __init__(self,dropout_prob):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=dropout_prob)            
        )
        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=dropout_prob)
        )
        self.trans2 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
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
             )
        
        self.pool1= nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(14, 10) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.conv3(x)
        x = self.trans3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.flat(x)
        x = self.fc(x)
        x =  F.log_softmax(x ,dim=1)
        return x

def print_model_summary(dropout_prob, inputsize):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(dropout_prob=dropout_prob).to(device)
    summary(model, input_size=inputsize)