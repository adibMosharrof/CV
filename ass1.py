import tensorflow as tf
import time
import pandas as pd
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import platform
from subprocess import check_output
import socket
import re
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def predict(x_train,x_test, y_train, distance_func):
    predictions = []  

    # loop over all test samples
    for x_t in x_test:
      # array of distances between current test and all training samples
      distances = np.sum(distance_func(x_train - x_t), axis=1)

      # get the closest one
      min_index = np.argmin(distances)

      # add corresponding label
      predictions.append(y_train[min_index])

    return predictions

def kmeans():
    x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
    pred1 = predict(x_train, x_test, y_train, np.abs) 
    pred2 = predict(x_train, x_test, y_train, np.square) 
    
    accuracy1 = (pred1 == y_test).astype(float).mean().item()
    accuracy2 = (pred2 == y_test).astype(float).mean().item()
    
    print("K-NN on CIFAR dataset with L1 distance: test accuracy = {:.2f}".format(accuracy1*100))
    print("K-NN on CIFAR dataset with L2 distance: test accuracy = {:.2f}".format(accuracy2*100))
    a=1

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class LizardCnn(nn.Module):
    def __init__(self):
        super(LizardCnn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        # (1) input layer
        t = t
        
        # (2) hidden conv layer
        #Size changes from (3, 32, 32) to (6, 28, 28)
        t = self.conv1(t)
        t = F.relu(t)
        # size changes from (6, 28, 28) to (6, 14, 14)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (3) hidden conv layer
        # size changes from (6, 14, 14) to (12, 10, 10)
        t = self.conv2(t)
        t = F.relu(t)
        # size changes from (12, 10, 10) to (12, 5, 5)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (4) hidden linear layer
        # size changes to (300)
        t = t.reshape(-1, 12 * 5 * 5)
        # size changes from (300) to (120)
        t = self.fc1(t)
        t = F.relu(t)
        
        # (5) hidden linear layer
        # size changes from (120) to (60)
        t = self.fc2(t)
        t = F.relu(t)
        
        # (6) output layer
        # size changes from (60) to (10)
        t = self.out(t)
        
        return t

class SimpleCnn(nn.Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3, stride=1, padding=1)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Linear(18 * 16 * 16, 64)
        
        #64 input features, 10 output features for our 10 defined classes
        self.out = nn.Linear(64, 10)

    def forward(self, t):
        # (1) input layer
        t = t
        #Size changes from (3, 32, 32) to (18, 32, 32)
        t = F.relu(self.conv1(t))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (-1, 4608)
        t = t.reshape(-1, 18 * 16 *16)
        
        #Size changes from (1, 4608) to (1, 64)
        t = self.fc1(t)
        t = F.relu(t)
        
        #Size changes from (1, 64) to (1, 10)
        t = self.out(t)
        return(t)
    
class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(24 * 4 * 4, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,32)
        self.out = nn.Linear(32,10)

    def forward(self, t):
        t = t
        
        # (2) hidden conv layer
        #Size changes from (3, 32, 32) to (6, 32, 32)
        t = self.conv1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = F.relu(t)
        # size changes from (6, 32, 32) to (6, 16, 16)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (3) hidden conv layer
        #Size changes from (6, 16, 16) to (12, 16, 16)
        t = self.conv3(t)
        t = F.relu(t)
        t = self.conv4(t)
        t = F.relu(t)
        # size changes from (12, 16, 16) to (12, 8, 8)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (4) hidden conv layer
        #Size changes from (12, 8, 8) to (24, 8, 8)
        t = self.conv5(t)
        t = F.relu(t)
        t = self.conv6(t)
        t = F.relu(t)
        # size changes from (24, 8, 8) to (24, 4, 4)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (24, 4, 4) to (-1, 384)
        t = t.reshape(-1, 24 * 4 *4)
        
        #Size changes from (1, 384) to (1, 128)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        
        # size changes from (128) to (32)

        t = self.fc3(t)
        t = F.relu(t)

        t = self.out(t)
        return t
    
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Linear(3 * 32 * 32, 2000, bias=True)
        self.fc2 = nn.Linear(2000,800, bias=True)
        self.fc3 = nn.Linear(800, 200, bias=True)

        #64 input features, 10 output features for our 10 defined classes
        self.out = nn.Linear(200,10, bias=True)

    def forward(self, t):
        # (1) input layer
        t = t
        t = t.reshape(-1, 3*32*32)
        #Size changes from (3, 32, 32) to (18, 32, 32)
        t = F.relu(self.fc1(t))

        t = F.relu(self.fc2(t))
        
        t = F.relu(self.fc3(t))

        t = self.out(t)
        return t
        
def part2():
    #loading and transforming the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./cifar-10-batches-py-1', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar-10-batches-py-1', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
#     cnn = Vgg()
#     cnn = SimpleCnn()
#     cnn = LizardCnn()
    cnn = Linear()
    train(cnn,trainloader)
    dataiter = iter(testloader)

    outputs = cnn(images)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

def train(net, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

def part4():
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    print(model_ft)
    


def train_model(model, dataloaders, criterion, optimizer, num_epochs=2, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

"""
    The methods below have been taken from 
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py
    These methods help to load the cifar10 dataset
"""

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=490, num_validation=10, num_test=100):
    # Load the raw CIFAR-10 data
    cifar10_dir = './cifar-10-batches-py'
    
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test
"""
End of methods for loading the dataset
"""


if __name__ == '__main__':
#     kmeans()
    part2()

