import torch
from torchvision import datasets, transforms


def get_train_loader(**kwargs):
   
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomRotation((-5.0, 5.0), fill=(1,)),                          
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),**kwargs)
    return train_loader

def get_test_loader(**kwargs):
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])), **kwargs)
    return test_loader
