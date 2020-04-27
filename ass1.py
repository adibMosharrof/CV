import tensorflow as tf
import time
import pandas as pd
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
# from pykeops.torch import LazyTensor
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
    env = get_env_name()
    if env == "local":
        cifar10_dir = 'E:/Study/CV/cifar-10-batches-py'
    else:
        cifar10_dir = '/home/adibm/uoml/adib/CV/cifar-10-batches-py'
        
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
def get_env_name():
    hostname = socket.gethostname()
    if hostname == "LAPTOP-DNMQC5VO":
        return "local"
    elif "uomlmedifor" in hostname:
        return "openstack" 
    elif re.search("(?<=n).\d+", hostname) is not None:
        return "talapas" 
    else:
        ValueError(f"Need to configure the new hostname {hostname}")




def predict(x_train,x_test, y_train, distance_func):
    predictions = []  # placeholder for N labels

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
    K=3
    start = time.time()
    #X_i = LazyTensor(x_test[:, None, :])  # (10000, 1, 784) test set
    #X_j = LazyTensor(x_train[None, :, :])  # (1, 60000, 784) train set
    # D_ij = ((X_i - X_j) ** 2).sum(-1)  # (10000, 60000) symbolic matrix of squared L2 distances
    pred1 = predict(x_train, x_test, y_train, np.abs) 
    pred2 = predict(x_train, x_test, y_train, np.square) 
    
    # ind_knn = D_ij.argKmin(K, dim=1)  # Samples <-> Dataset, (N_test, K)
    # lab_knn = y_train[ind_knn]  # (N_test, K) array of integers in [0,9]
    # y_knn, _ = pred.mode()   # Compute the most likely label
    end = time.time()
    
    accuracy1 = (pred1 == y_test).astype(float).mean().item()
    accuracy2 = (pred2 == y_test).astype(float).mean().item()
    # accuracy1 = 1.0-error1
    # accuracy2 = 1.0-error2
    time  = end - start
    
    print("K-NN on CIFAR dataset with L1 distance: test accuracy = {:.2f}% in {:.2f}s.".format(accuracy1*100, time))
    print("K-NN on CIFAR dataset with L2 distance: test accuracy = {:.2f}% in {:.2f}s.".format(accuracy2*100, time))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def part2():
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
#     imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    cnn = Cnn()
    train(cnn,trainloader)
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
#     imshow(torchvision.utils.make_grid(images))
#     print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

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

def init():
    x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
    
    
    print('Train data shape: ', x_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', x_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)

if __name__ == '__main__':
    init()
    #kmeans()
    part2()

