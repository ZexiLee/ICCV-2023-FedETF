
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import random
from torch.backends import cudnn
import math
from pyhessian import hessian
from torch.optim import Optimizer
from models_dict import densenet, resnet, cnn
import copy

##############################################################################
# Tools
##############################################################################
def compute_classifier_cosine(args, client_nodes):
    num_clients = len(client_nodes)
    classifier_weights = []
    for i in range(num_clients):
        classifier_weights.append(client_nodes[i].model.state_dict()['linear_head.weight'].view(-1))
    avg_cosine = (sum([sum(i) for i in get_cosGrad_matrix(classifier_weights)]) - num_clients) / (num_clients**2 - num_clients)
    return avg_cosine

def client_fednh_compute_proto(args, node):
    node.model.eval()
    train_loader = node.local_data
    with torch.no_grad():
        agg_protos_label = {}
        for idx, (data, target) in enumerate(train_loader):

            data, labels = data.cuda(), target.cuda()
            _, _, features = node.model(data)
            # update proto
            for i in range(len(labels)):
                if labels[i].cpu().item() in agg_protos_label:
                    agg_protos_label[labels[i].cpu().item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].cpu().item()] = [features[i, :]]
        protos = agg_protos_label
        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]
    node.agg_protos = protos

    return


def compute_proto_consine(args, client_nodes):
    num_clients = len(client_nodes)
    local_protos_list = {}
    for i in range(num_clients):
        client_fednh_compute_proto(args, client_nodes[i])
        for label in client_nodes[i].agg_protos:
            if label in local_protos_list:
                local_protos_list[label].append(client_nodes[i].agg_protos[label])
            else:
                local_protos_list[label] = [client_nodes[i].agg_protos[label]]
    label_cosines = []
    for label in range(client_nodes[0].num_classes):
        num_protos = len(local_protos_list[label])
        label_cosines.append((sum([sum(i) for i in get_cosGrad_matrix(local_protos_list[label])]) - num_protos) / (num_protos**2 - num_protos))
    avg_cosine = sum(label_cosines)/len(label_cosines)
    return avg_cosine

def compute_globalmodel_neural_collapse(args, central_node):

    client_fednh_compute_proto(args, central_node)
    protos = copy.deepcopy(central_node.agg_protos)
    
    central_node.model.eval()
    train_loader = central_node.local_data
    cosine_list = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(train_loader):

            data, labels = data.cuda(), target.cuda()
            _, _, features = central_node.model(data)
            # update proto
            for i in range(len(labels)):
                cosine_list.append(cos(features[i, :], protos[labels[i].cpu().item()]))

    avg_cosine = sum(cosine_list)/len(cosine_list)

    return avg_cosine

def set_server_method(args):
    if args.client_method in ['local_train', 'fedrod', 'fedprox', 'ditto', 'fedetf']:
        args.server_method = 'fedavg'
    else:
        args.server_method = args.client_method
    return args


class Model(nn.Module):
    """For classification problem"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_params(self):
        return self.state_dict()

    def get_gradients(self, dataloader):
        raise NotImplementedError


def set_params(model, model_state_dict, exclude_keys=set()):
    """
        Reference: Be careful with the state_dict[key].
        https://discuss.pytorch.org/t/how-to-copy-a-modified-state-dict-into-a-models-state-dict/64828/4.
    """
    with torch.no_grad():
        for key in model_state_dict.keys():
            if key not in exclude_keys:
                model.state_dict()[key].copy_(model_state_dict[key])
    return model

def freeze_layers(model, layers_to_freeze):
    for name, p in model.named_parameters():
        try:
            if name in layers_to_freeze:
                p.requires_grad = False
            else:
                p.requires_grad = True
        except:
            pass
    return model

class ModelWrapper(Model):
    def __init__(self, base, head, config):
        """
            head and base should be nn.module
        """
        super(ModelWrapper, self).__init__(config)

        self.base = base
        self.head = head

    def forward(self, x, return_embedding):
        feature_embedding = self.base(x)
        out = self.head(feature_embedding)
        if return_embedding:
            return feature_embedding, out
        else:
            return out


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)

def softmax_fuct(lrs):
    '''
    lrs is dict as {0:3, 1:3, 2:4}
    '''
    exp_cache = []
    softmax_lrs = {}
    for i in range(len(lrs)):
        exp_cache.append(math.exp(lrs[i]))
    
    for i in range(len(lrs)):
        softmax_lrs[i] = exp_cache[i]/sum(exp_cache)
    
    return softmax_lrs

def cos(x, y):
    fuct = nn.CosineSimilarity(dim=0)
    result = fuct(x, y)
    result = result.detach().cpu().numpy().tolist()
    return result

def get_cosGrad_matrix(gradients):
    client_num = len(gradients)
    matrix = [[0.0 for _ in range(client_num)] for _ in range(client_num)]

    for i in range(client_num):
        for j in range(client_num):
            if matrix[j][i] != 0.0:
                matrix[i][j] = matrix[j][i]
            else:
                matrix[i][j] = cos(gradients[i], gradients[j])
    
    return matrix

def model_parameter_vector(args, model):
    param = [p.view(-1) for p in model.parameters()]
    # vector = torch.concat(param, dim=0)
    vector = torch.cat(param, dim=0)
    return vector

##############################################################################
# Initialization function
##############################################################################

def init_model(model_type, args):
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100

    if model_type == 'CNN':
        if args.dataset == 'cifar10':
            model = cnn.CNNCifar10()
        else:
            model = cnn.CNNCifar100()
    # resnet18 for imagenet
    elif model_type == 'ResNet18':
        model = resnet.ResNet18()
    elif model_type == 'ResNet20':
        model = resnet.ResNet20(num_classes)
    elif model_type == 'ResNet56':
        model = resnet.ResNet56(num_classes)
    elif model_type == 'ResNet110':
        model = resnet.ResNet110(num_classes)
    elif model_type == 'WRN56_2':
        model = resnet.WRN56_2(num_classes)
    elif model_type == 'WRN56_4':
        model = resnet.WRN56_4(num_classes)
    elif model_type == 'WRN56_8':
        model = resnet.WRN56_8(num_classes)
    elif model_type == 'DenseNet121':
        model = densenet.DenseNet121(num_classes)
    elif model_type == 'DenseNet169':
        model = densenet.DenseNet169(num_classes)
    elif model_type == 'DenseNet201':
        model = densenet.DenseNet201(num_classes)
    elif model_type == 'MLP':
        model = cnn.MLP()
    elif model_type == 'LeNet5':
        model = cnn.LeNet5() 

    return model

def init_optimizer(num_id, model, args):
    optimizer = []
    if num_id > -1 and args.client_method == 'fedprox':
        optimizer = PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

##############################################################################
# Training function
##############################################################################

def generate_matchlist(client_node, ratio = 0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list

def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if rounds != 0:
        args.lr *= 0.99 #0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    # print('Learning rate={:.4f}'.format(args.lr))


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                # g = g.cuda()
                if p.grad != None:
                    d_p = p.grad.data + group['mu'] * (p.data - g.data)
                    p.data.add_(d_p, alpha=-group['lr'])

##############################################################################
# Validation function
##############################################################################

def validate(args, node, which_dataset = 'validate'):
    '''
    Generally, 'validate' refers to the local datasets of clients and 'local' refers to the server's testset.
    '''
    node.model.cuda().eval() 
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            feature, logit, out = node.model(data)
            if 'fedetf' in args.client_method:
                output = torch.matmul(feature, node.model.proto_classifier.proto)
            elif args.client_method == 'fedrod':
                if which_dataset == 'validate':
                    # client local data
                    logit_p = node.p_head(out)
                    output = logit + logit_p
                else:
                    # server-side testset
                    output = logit
            elif args.client_method == 'fednh':
                feature = out
                feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                feature_embedding = torch.div(feature, feature_norm)
                normalized_prototype = node.prototype
                logits = torch.matmul(feature_embedding, normalized_prototype.T)
                output = logits
            elif args.client_method == 'ditto':
                if which_dataset == 'validate':
                    # client local data
                    _, logit, _ = node.p_model(data)
                output = logit
            else:
                output = logit

            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset) * 100
    return acc

def testloss(args, node, which_dataset = 'validate'):
    node.model.cuda().eval()  
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            _, output, _ = node.model(data)
            loss_local =  F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
    loss_value = sum(loss)/len(loss)
    return loss_value