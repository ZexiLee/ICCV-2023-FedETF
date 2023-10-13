

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
# from Data import DatasetSplit
from datasets import DatasetSplit
from utils import init_model, PerturbedGradientDescent
from utils import init_optimizer, model_parameter_vector

class Node(object):

    def __init__(self, num_id, local_data, train_set, args):
        self.num_id = num_id
        self.args = args
        self.node_num = self.args.node_num
        if num_id == -1:
            self.valid_ratio = args.server_valid_ratio
        else:
            self.valid_ratio = args.client_valid_ratio

        if self.args.dataset in ['cifar10', 'fmnist']:
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100
        elif self.args.dataset == 'tinyimagenet':
            self.num_classes = 200

        if args.iid == 1 or num_id == -1:
            # for the server, use the validate_set as the training data, and use local_data for testing
            self.local_data, self.validate_set = self.train_val_split_forServer(local_data.indices, train_set, self.valid_ratio, self.num_classes)
        else:
            self.local_data, self.validate_set = self.train_val_split(local_data, train_set, self.valid_ratio)
            # generate sample_per_class
            self.sample_per_class = self.generate_sample_per_class(self.local_data)
        

        self.model = init_model(self.args.local_model, self.args).cuda()
        self.optimizer = init_optimizer(self.num_id, self.model, args)
        
        # node init for feddyn
        if args.client_method == 'feddyn':
            self.old_grad = None
            self.old_grad = copy.deepcopy(self.model)
            self.old_grad = model_parameter_vector(args, self.old_grad)
            self.old_grad = torch.zeros_like(self.old_grad)
        if 'feddyn' in args.server_method:
            self.server_state = copy.deepcopy(self.model)
            for param in self.server_state.parameters():
                param.data = torch.zeros_like(param.data)
        
        # p_head for fedrod and fedrep
        if args.client_method == 'fedrod':
            self.p_head = copy.deepcopy(self.model.linear_head)
        
        # prototype for fednh
        if args.client_method == 'fednh':
            # # the default dim is 64
            # temp = nn.Linear(64, self.num_classes, bias=False).state_dict()['weight']
            # for imagenet, the default dim is 512
            # for fmnist, it's 
            if args.dataset == 'fmnist':
                dim = 32*7*7
            elif args.dataset == 'tinyimagenet':
                dim = 512
            else:
                dim = 64
            temp = nn.Linear(dim, self.num_classes, bias=False).state_dict()['weight']
            self.prototype = nn.Parameter(temp, requires_grad=False)
            self.scaling = torch.nn.Parameter(torch.tensor([self.args.scaling_train]), requires_grad=False).cuda()
            m, n = self.prototype.shape
            self.prototype.data = torch.nn.init.orthogonal_(torch.rand(m, n)).cuda()

            # local prototype buffer
            self.agg_protos = None

        # prototype for fedproto
        elif args.client_method == 'fedproto':
            self.prototype = None
            self.agg_protos = None

        # feature means vars for ccvr
        elif args.client_method == 'ccvr':
            self.feature_meanvar = None

        
        # p_model and p_optimizer for ditto
        if args.client_method == 'ditto':
            self.p_model = init_model(self.args.local_model, self.args).cuda()
            self.p_optimizer = PerturbedGradientDescent(self.p_model.parameters(), lr=args.lr, mu=args.mu)
        
        # head_key for fedrep
        if args.client_method in ['fedrep',  'ccvr']:
            self.head_key = set([name for name in self.model.state_dict().keys() if 'head' in name])
            self.base_key = set([key for key in self.model.state_dict().keys()
                                       if key not in self.head_key])



    def train_val_split(self, idxs, train_set, valid_ratio): 

        np.random.shuffle(idxs)

        validate_size = valid_ratio * len(idxs)
        # print(len(idxs))

        idxs_test = idxs[:int(validate_size)]
        idxs_train = idxs[int(validate_size):]

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train),
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=True)

        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.validate_batchsize,  num_workers=0, shuffle=True)
        

        return train_loader, test_loader

    def train_val_split_forServer(self, idxs, train_set, valid_ratio, num_classes=10): # local data index, trainset

        np.random.shuffle(idxs)

        validate_size = valid_ratio * len(idxs)

        # generate proxy dataset with balanced classes
        idxs_test = []
        test_class_count = [int(validate_size)/num_classes for _ in range(num_classes)]
        k = 0
        while sum(test_class_count) != 0:
            if test_class_count[train_set[idxs[k]][1]] > 0:
                idxs_test.append(idxs[k])
                test_class_count[train_set[idxs[k]][1]] -= 1
            else: 
                pass
            k += 1
        label_list = []
        for k in idxs_test:
            label_list.append(train_set[k][1])

        idxs_train = [idx for idx in idxs if idx not in idxs_test]

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train),
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=True)
        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.validate_batchsize,  num_workers=0, shuffle=True)

        return train_loader, test_loader

    def generate_sample_per_class(self, local_data):
        sample_per_class = torch.tensor([0 for _ in range(self.num_classes)])

        for idx, (data, target) in enumerate(local_data):
            sample_per_class += torch.tensor([sum(target==i) for i in range(self.num_classes)])

        sample_per_class = torch.where(sample_per_class > 0, sample_per_class, 1)

        return sample_per_class

    def compute_sum_proto_cos(self):
        train_loader = self.local_data  # iid
        cos_per_label = [[] for _ in range(self.num_classes)]
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                proto = self.model.proto_classifier(target)
                feature, _, _ = self.model(data)
                proto_cos = torch.bmm(feature.unsqueeze(1), proto.unsqueeze(2)).view(-1) 

                for i, label in enumerate(target):
                    cos_per_label[label].append(proto_cos[i])

        cos_per_label = [sum(item)/len(item) if item != [] else 0 for item in cos_per_label]
        cos_per_label = torch.tensor(cos_per_label)

        return cos_per_label.sum()