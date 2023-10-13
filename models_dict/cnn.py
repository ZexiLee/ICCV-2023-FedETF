import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
from contextlib import contextmanager
import torch
import torch.nn as nn
import torchvision
from six import add_metaclass
from torch.nn import init
import copy
import math
from .proto_classifiers import Proto_Classifier

class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, 3) #0
        self.maxpool = nn.MaxPool2d(2, 2)
        # self.layers.append(nn.MaxPool2d(2, 2)) # 1
        self.layer2 = nn.Conv2d(32, 64, 3) # 2
        self.layer3 = nn.Conv2d(64, 64, 3) #3
        self.layer4 = nn.Linear(64 * 4 * 4, 64)#4
        # proto classifier
        self.linear_proto = nn.Linear(64, 64)
        self.proto_classifier = Proto_Classifier(64, 10)
        self.scaling_train = torch.nn.Parameter(torch.tensor(10.0))

        # linear classifier
        # set bias as false only for motivation figure
        self.linear_head = nn.Linear(64, 10)

    def forward(self, x):
        x = self.maxpool(F.relu(self.layer1(x)))
        x = self.maxpool(F.relu(self.layer2(x)))
        x = F.relu(self.layer3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.layer4(x))
        out = x

        # proto classifier: generate normalized features
        feature = self.linear_proto(x)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)

        # linear classifier
        logit = self.linear_head(x)

        return feature, logit, out


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        # proto classifier for FedETF
        self.linear_proto = nn.Linear(84, 10)
        self.proto_classifier = Proto_Classifier(10, 10)
        # self.proto_classifier = Proto_Classifier(64, num_classes)
        self.scaling_train = torch.nn.Parameter(torch.tensor(1.0))

        # linear classifier, also the g_head in FedRoD
        self.linear_head = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = x
        # proto classifier: generate normalized features
        feature = self.linear_proto(x)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)
        # linear classifier
        logit = self.linear_head(x)
        return feature, logit, out 

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
