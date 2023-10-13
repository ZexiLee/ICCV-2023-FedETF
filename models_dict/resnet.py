import torch
import torch.nn as nn
import torch.nn.functional as F
from .proto_classifiers import Proto_Classifier

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_noshortcut(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_noshortcut(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        return out

# for imagenet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # proto classifier for FedETF
        self.linear_proto = nn.Linear(512*block.expansion, num_classes)
        self.proto_classifier = Proto_Classifier(num_classes, num_classes)
        # self.proto_classifier = Proto_Classifier(64, num_classes)
        self.scaling_train = torch.nn.Parameter(torch.tensor(1.0))

        # linear classifier, also the g_head in FedRoD
        self.linear_head = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        # proto classifier: generate normalized features
        feature = self.linear_proto(out)
        # feature = out

        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)
        # linear classifier
        logit = self.linear_head(out)
        return feature, logit, out # feature for FedETF, logit for others, feature for others



class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16

        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # proto classifier for FedETF
        self.linear_proto = nn.Linear(64*block.expansion, num_classes)
        self.proto_classifier = Proto_Classifier(num_classes, num_classes)
        # self.proto_classifier = Proto_Classifier(64, num_classes)
        self.scaling_train = torch.nn.Parameter(torch.tensor(1.0))

        # linear classifier, also the g_head in FedRoD
        self.linear_head = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        # proto classifier: generate normalized features
        feature = self.linear_proto(out)
        # feature = out

        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)
        # linear classifier
        logit = self.linear_head(out)
        return feature, logit, out # feature for FedETF, logit for others, feature for others


class WResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(WResNet_cifar, self).__init__()
        self.in_planes = 16*k

        self.conv1 = nn.Conv2d(3, 16*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*k)
        self.layer1 = self._make_layer(block, 16*k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*k, num_blocks[2], stride=2)
        # proto classifier
        self.linear_proto = nn.Linear(64*k*block.expansion, num_classes)
        self.proto_classifier = Proto_Classifier(num_classes, num_classes)
        self.scaling_train = torch.nn.Parameter(torch.tensor(1.0))
        # linear classifier
        self.linear = nn.Linear(64*k*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        # proto classifier: generate normalized features
        feature = self.linear_proto(out)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature = torch.div(feature, feature_norm)
        # linear classifier
        logit = self.linear(out)
        return feature, logit, out

# ImageNet models
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

# def ResNet18_noshort():
#     return ResNet(BasicBlock_noshortcut, [2,2,2,2])

# def ResNet34():
#     return ResNet(BasicBlock, [3,4,6,3])

# def ResNet34_noshort():
#     return ResNet(BasicBlock_noshortcut, [3,4,6,3])

# def ResNet50():
#     return ResNet(Bottleneck, [3,4,6,3])

# def ResNet50_noshort():
#     return ResNet(Bottleneck_noshortcut, [3,4,6,3])

# def ResNet101():
#     return ResNet(Bottleneck, [3,4,23,3])

# def ResNet101_noshort():
#     return ResNet(Bottleneck_noshortcut, [3,4,23,3])

# def ResNet152():
#     return ResNet(Bottleneck, [3,8,36,3])

# def ResNet152_noshort():
#     return ResNet(Bottleneck_noshortcut, [3,8,36,3])

# CIFAR-10/100 models
def ResNet20(num_classes):
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)

def ResNet20_noshort(num_classes):
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n], num_classes)

def ResNet32(num_classes):
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)

def ResNet32_noshort(num_classes):
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n], num_classes)

def ResNet44_noshort(num_classes):
    depth = 44
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n], num_classes)

def ResNet50_16_noshort(num_classes):
    depth = 50
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n], num_classes)

def ResNet56(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)

def ResNet56_noshort(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n], num_classes)

def ResNet110(num_classes):
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes)

def ResNet110_noshort(num_classes):
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n], num_classes)

def WRN56_2(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 2, num_classes)

def WRN56_4(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 4, num_classes)

def WRN56_8(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 8, num_classes)

def WRN56_2_noshort(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 2, num_classes)

def WRN56_4_noshort(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 4, num_classes)

def WRN56_8_noshort(num_classes):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 8, num_classes)

def WRN110_2_noshort(num_classes):
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 2, num_classes)

def WRN110_4_noshort(num_classes):
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 4, num_classes)