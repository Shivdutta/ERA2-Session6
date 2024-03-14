# ERA 2 Session 6 : Backpropagation Part 2 

### Introduction

Training a model using MNIST data is a straightforward process. Essentially, it involves creating a classification model designed to analyze an image and accurately predict the handwritten digit it contains. However, to inject a bit of enjoyment, we've introduced certain constraints. Before delving into the code, let's first familiarize ourselves with the model's input, output, and architecture. This is how the structure of the README has been organized.

### Input
MNIST data comprises images depicting handwritten digits.

![MINIST](images/mnist.png)

- Constraints    Keep the parameters under 20,000.
- Execute the model precisely for 19 epochs.
- Incorporate Batch Normalization, FC Layer, Dropout, and GAP.
- Utilize a 1x1 kernel to showcase the efficacy of deep learning without imposing excessive strain on the GPU, essentially streamlining the process.
    
### Network 
- The network comprises approximately four sets of convolutional blocks, each containing 2 to 3 convolution blocks, followed by transition block (containing Maxpool layers), Global Average Pooling (GAP) layer, and Fully Connected (FC) layer.

#### Typical Structure 
- Convolution Layer of 3 x 3 Kernel with Padding of 1 
- ReLu - Activation Function
- Batch Normalization - Regularization Parameter
- Dropout             - Regularization Parameter 
- Maxpool after 3 Layers 

- Example 
self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=dropout_value)
        )

- Global Average Pooling: A pooling operation that computes the average value of each feature map across its entire spatial dimensions, resulting in a single value per feature map.

- Flatten: The process of transforming a multidimensional tensor (2D or 3D) into a one-dimensional tensor before sending to fully connected layers.

- Fully Connected Linear Layer: A layer in a neural network where each neuron is connected to every neuron in the previous layer, performing a linear transformation on the input data. 

#### Constraint
- The Maxpool layers are positioned to achieve a receptive field of 7x7 or 9x9, with a preference for 11x11 for more complex datasets like ImageNet. However, given the simplicity of the MNIST dataset, a 7x7 receptive field is deemed sufficient.

- Remove ReLu, Batch Normalization (BN), or Dropout in the final layers of the network.

- Use for minimal parameters, approaching the lower limit of 20,000.

- Apply a Dropout rate best suited

- Implement Batch Normalization for improved training stability.

- Utilize a batch size of 128 during training.

#### Output
- Achieve an accuracy of greater than or equal to 99.4% within 20 epochs.

#### Training Details
- The validation accuracy of 99.38 is acheieved in epoch 16 to 18 and is consistent and finally at 19 epoch it reached to 99.46
The modern archtecture(pyramid) is followed and each block has same kernal size except in last one which done to reduce the paramters and receptive field size. The network comprises approximately four sets of convolutional blocks, each containing 2 to 3 convolution blocks, followed by a transition block

###### Training logs 
Currently Executing Epoch: 1
Loss=0.13996659219264984 Batch_id=468 Accuracy=87.95: 100%|██████████| 469/469 [00:11<00:00, 41.17it/s]

Test set: Average loss: 0.0686, Accuracy: 9784/10000 (97.84%)

Currently Executing Epoch: 2
Loss=0.07255340367555618 Batch_id=468 Accuracy=97.64: 100%|██████████| 469/469 [00:11<00:00, 40.16it/s]

Test set: Average loss: 0.0533, Accuracy: 9838/10000 (98.38%)

Currently Executing Epoch: 3
Loss=0.020857753232121468 Batch_id=468 Accuracy=98.24: 100%|██████████| 469/469 [00:11<00:00, 40.65it/s]

Test set: Average loss: 0.0377, Accuracy: 9873/10000 (98.73%)

Currently Executing Epoch: 4
Loss=0.02174842357635498 Batch_id=468 Accuracy=98.53: 100%|██████████| 469/469 [00:11<00:00, 40.20it/s]

Test set: Average loss: 0.0317, Accuracy: 9892/10000 (98.92%)

Currently Executing Epoch: 5
Loss=0.02022217959165573 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:11<00:00, 40.48it/s]

Test set: Average loss: 0.0320, Accuracy: 9900/10000 (99.00%)

Currently Executing Epoch: 6
Loss=0.0736260637640953 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:11<00:00, 40.31it/s]

Test set: Average loss: 0.0295, Accuracy: 9897/10000 (98.97%)

Currently Executing Epoch: 7
Loss=0.04806552827358246 Batch_id=468 Accuracy=99.02: 100%|██████████| 469/469 [00:11<00:00, 39.53it/s]

Test set: Average loss: 0.0268, Accuracy: 9915/10000 (99.15%)

Currently Executing Epoch: 8
Loss=0.013765481300652027 Batch_id=468 Accuracy=99.08: 100%|██████████| 469/469 [00:11<00:00, 39.95it/s]

Test set: Average loss: 0.0269, Accuracy: 9925/10000 (99.25%)

Currently Executing Epoch: 9
Loss=0.022269509732723236 Batch_id=468 Accuracy=99.09: 100%|██████████| 469/469 [00:11<00:00, 39.91it/s]

Test set: Average loss: 0.0301, Accuracy: 9904/10000 (99.04%)

Currently Executing Epoch: 10
Loss=0.03125382959842682 Batch_id=468 Accuracy=99.15: 100%|██████████| 469/469 [00:11<00:00, 40.63it/s]

Test set: Average loss: 0.0268, Accuracy: 9918/10000 (99.18%)

Currently Executing Epoch: 11
Loss=0.01220733392983675 Batch_id=468 Accuracy=99.22: 100%|██████████| 469/469 [00:11<00:00, 39.76it/s]

Test set: Average loss: 0.0253, Accuracy: 9921/10000 (99.21%)

Currently Executing Epoch: 12
Loss=0.02853059209883213 Batch_id=468 Accuracy=99.20: 100%|██████████| 469/469 [00:12<00:00, 38.97it/s]

Test set: Average loss: 0.0223, Accuracy: 9916/10000 (99.16%)

Currently Executing Epoch: 13
Loss=0.004551796242594719 Batch_id=468 Accuracy=99.29: 100%|██████████| 469/469 [00:11<00:00, 40.06it/s]

Test set: Average loss: 0.0220, Accuracy: 9925/10000 (99.25%)

Currently Executing Epoch: 14
Loss=0.008316882885992527 Batch_id=468 Accuracy=99.32: 100%|██████████| 469/469 [00:11<00:00, 39.58it/s]

Test set: Average loss: 0.0219, Accuracy: 9932/10000 (99.32%)

Currently Executing Epoch: 15
Loss=0.008117679506540298 Batch_id=468 Accuracy=99.31: 100%|██████████| 469/469 [00:11<00:00, 39.38it/s]

Test set: Average loss: 0.0251, Accuracy: 9920/10000 (99.20%)

Currently Executing Epoch: 16
Loss=0.016682356595993042 Batch_id=468 Accuracy=99.52: 100%|██████████| 469/469 [00:11<00:00, 40.27it/s]

Test set: Average loss: 0.0199, Accuracy: 9938/10000 (99.38%)

Currently Executing Epoch: 17
Loss=0.02587145008146763 Batch_id=468 Accuracy=99.61: 100%|██████████| 469/469 [00:11<00:00, 39.86it/s]

Test set: Average loss: 0.0191, Accuracy: 9939/10000 (99.39%)

Currently Executing Epoch: 18
Loss=0.0013698586262762547 Batch_id=468 Accuracy=99.65: 100%|██████████| 469/469 [00:11<00:00, 40.38it/s]

Test set: Average loss: 0.0193, Accuracy: 9938/10000 (99.38%)

Currently Executing Epoch: 19
Loss=0.020460905507206917 Batch_id=468 Accuracy=99.68: 100%|██████████| 469/469 [00:11<00:00, 39.79it/s]

Test set: Average loss: 0.0191, Accuracy: 9944/10000 (99.44%)

The network is trained with Batch Normalization and Dropout and accuracy is greater than 99.4 %. The loss and accuracy curve is almost smooth except little spikes in validation data but eventually it is smooth at the end

![accuracy](images/accuracy.png)   ![loss](images/loss.png)

![accuracy_loss](images/accuracy_loss.png)

As we go in future session, the architecture will be improved with new learning.
Thank you.
