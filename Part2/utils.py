"""
Module Imports for Data Visualization, PyTorch, and Utility Libraries.

This module includes import statements for essential libraries related to data visualization,
PyTorch, and utility functions. It covers modules for creating plots (matplotlib.pyplot),
customizing plot ticks (matplotlib.ticker), deep learning (torch), model summary visualization
(torchsummary), and training progress visualization (tqdm).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F

def plot_loss(train_loss_list, test_loss_list):
    """
    Plot Training and Testing Loss Over Epochs.

    This function generates a line plot to visualize the training and testing loss over epochs.

    Parameters:
    - train_loss_list (list): List containing training loss values for each epoch.
    - test_loss_list (list): List containing testing loss values for each epoch.
    """
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(train_loss_list, label="Train Loss")
    axs.plot(test_loss_list, label="Test Loss")
    axs.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_accuracy(train_accuracy_list, test_accuracy_list):
    """
    Plot Training and Testing Accuracy Over Epochs.

    This function generates a line plot to visualize the training and testing accuracy over epochs.

    Parameters:
    - train_accuracy_list (list): List containing training accuracy values for each epoch.
    - test_accuracy_list (list): List containing testing accuracy values for each epoch.
    """
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(train_accuracy_list, label="Train Accuracy")
    axs.plot(test_accuracy_list, label="Test Accuracy")
    axs.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

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
    summary(network, input_size=input_size)

def train_model(model, device, train_loader, optimizer, epoch,train_losses,train_acc):
    """
    Train PyTorch Model on Training Data.

    This function trains a PyTorch model on the provided training data using the specified optimizer.
    It computes and logs the training loss and accuracy for each epoch.

    Parameters:
    - model (torch.nn.Module): PyTorch model to be trained.
    - device (torch.device): Device on which the model is loaded (e.g., "cuda" or "cpu").
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - optimizer (torch.optim.Optimizer): Optimizer used for updating model parameters during training.
    - epoch (int): Current epoch number.
    - train_losses (list): List to store training losses over epochs.
    - train_acc (list): List to store training accuracies over epochs.

    Returns:
    - train_losses (list): Updated list of training losses over epochs.
    - train_acc (list): Updated list of training accuracies over epochs.
    """  
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    loss_list = []
    acc_list = []
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        tensor_on_cpu = loss.cpu()
        numpy_array = tensor_on_cpu.detach().numpy()
        
        loss_list.append(numpy_array)
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm    
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        acc_list.append(100*correct/processed)
    train_losses.append(sum(loss_list)/len(loss_list))
    train_acc.append(sum(acc_list)/len(acc_list))

    return train_losses,train_acc

def test_model(model, device, test_loader,test_losses,test_acc):
    """
    Evaluate PyTorch Model on Test Data.

    This function evaluates a PyTorch model on the provided test data and computes test loss and accuracy.

    Parameters:
    - model (torch.nn.Module): PyTorch model to be evaluated.
    - device (torch.device): Device on which the model is loaded (e.g., "cuda" or "cpu").
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - test_losses (list): List to store test losses.
    - test_acc (list): List to store test accuracies.

    Returns:
    - test_losses (list): Updated list of test losses.
    - test_acc (list): Updated list of test accuracies.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    
    return test_losses,test_acc


def display_loss_and_accuracies(train_losses: list,
                                train_acc: list,
                                test_losses: list,
                                test_acc: list,
                                plot_size: tuple = (10, 10)) :
    """
    Function to display training and test information(losses and accuracies)
    :param train_losses: List containing training loss of each epoch
    :param train_acc: List containing training accuracy of each epoch
    :param test_losses: List containing test loss of each epoch
    :param test_acc: List containing test accuracy of each epoch
    :param plot_size: Size of the plot
    """
    # Create a plot of 2x2 of size
    fig, axs = plt.subplots(2, 2, figsize=plot_size)

    # Plot the training loss and accuracy for each epoch
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")

    # Plot the test loss and accuracy for each epoch
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
