import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.backends import cudnn
from random import sample
import math
import torch.optim as optim
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import init_model, freeze_layers, set_params
from datasets import TensorDataset

##############################################################################
# General server function
##############################################################################

def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    local_protos_list = {}

    for idx in select_list:
        if args.server_method != 'fedproto':
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
        if args.server_method in ['fednh', 'fedproto']:
            local_protos_list[idx] = copy.deepcopy(client_nodes[idx].agg_protos)
        elif args.server_method == 'ccvr':
            local_protos_list[idx] = copy.deepcopy(client_nodes[idx].feature_meanvar)
            
    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]

    return agg_weights, client_params, local_protos_list




##############################################################################
# Baselines function (FedAvg, FedDF, FedBE, Finetune, etc.)
##############################################################################

def Server_update(args, central_node, client_nodes, select_list, size_weights):
    '''
    server update functions for baselines
    '''

    # receive the local models from clients
    agg_weights, client_params, local_protos_list = receive_client_models(args, client_nodes, select_list, size_weights)

    # update the global model
    if args.server_method == 'fedavg':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)

    elif args.server_method == 'fednh':
        avg_global_param = fedavg(client_params, agg_weights)
        global_proto = fednh(local_protos_list, central_node)
        central_node.model.load_state_dict(avg_global_param)
        central_node.prototype.data = copy.deepcopy(global_proto)

    elif args.server_method == 'fedproto':
        global_proto = fedproto(local_protos_list, central_node)
        central_node.prototype = copy.deepcopy(global_proto)

    elif args.server_method == 'feddf':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
        central_node = feddf(args, central_node, client_nodes, select_list)

    elif args.server_method == 'fedbe':
        prev_global_param = copy.deepcopy(central_node.model.state_dict())
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
        central_node = fedbe(args, prev_global_param, central_node, client_nodes, select_list)

    elif args.server_method == 'finetune':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
        central_node = server_finetune(args, central_node)

    elif args.server_method == 'feddyn':
        central_node = feddyn(args, central_node, agg_weights, client_nodes, select_list)

    elif args.server_method == 'ccvr':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
        central_node = ccvr(args, central_node, local_protos_list)

    elif args.server_method == 'fedrep':
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model = set_params(central_node.model, avg_global_param, central_node.head_key)

    else:
        raise ValueError('Undefined server method...')

    return central_node

def server_finetune(args, central_node):
    central_node.model.train()
    for epoch in range(args.server_epochs): 
        # the training data is the small dataset on the server
        train_loader = central_node.validate_set 

        for _, (data, target) in enumerate(train_loader):

            central_node.optimizer.zero_grad()
            # train model
            data, target = data.cuda(), target.cuda()

            output = central_node.model(data)

            # compute losses according to the weights
            loss =  F.cross_entropy(output, target)
            loss.backward()
            central_node.optimizer.step()

    return central_node

def fedavg(parameters, list_nums_local_data):
    fedavg_global_params = copy.deepcopy(parameters[0])
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param
    return fedavg_global_params

# FedNH
def fednh(local_protos_list, central_node):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    init_global_prototypes = copy.deepcopy(central_node.prototype.data)
    global_prototypes = agg_protos_label

    # global: dict --- > tensor
    global_pro_list = []
    proto_label = global_prototypes.keys()
    for i in range(central_node.num_classes):
        if i in proto_label:
            value = global_prototypes[i][0]
            # global_pro_list.append(global_prototypes[i][0])
        else:
            value = torch.zeros(64).cuda()
        global_pro_list.append(value)
    global_pro_tensor = torch.stack(global_pro_list, dim=0)

    avg_prototype = F.normalize(global_pro_tensor, dim=1)
    # update prototype with moving average
    weight = 0.9
    global_proto = weight * init_global_prototypes + (1 - weight) * avg_prototype
    # print('agg weight:', weight)
    # normalize prototype again
    global_proto = F.normalize(global_proto, dim=1)

    return global_proto 

# FedProto
def fedproto(local_protos_list, central_node):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    global_prototypes = agg_protos_label

    # global: dict --- > tensor
    global_pro_list = []
    proto_label = global_prototypes.keys()
    for i in range(central_node.num_classes):
        if i in proto_label:
            value = global_prototypes[i][0]
            # global_pro_list.append(global_prototypes[i][0])
        else:
            value = torch.zeros(64).cuda()
        global_pro_list.append(value)
    global_pro_tensor = torch.stack(global_pro_list, dim=0)

    return global_pro_tensor


# FedDF
def divergence(student_logits, teacher_logits):
    divergence = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )  # forward KL
    return divergence

def feddf(args, central_node, client_nodes, select_list):
    # train and update
    central_node.model.cuda().train()
    nets = []
    for client_idx in select_list:
        client_nodes[client_idx].model.cuda().eval()
        nets.append(client_nodes[client_idx].model)

    for _ in range(args.server_epochs):
        train_loader = central_node.validate_set 

        for _, (data, target) in enumerate(train_loader):
            central_node.optimizer.zero_grad()
            # train model
            data, target = data.cuda(), target.cuda()

            output = central_node.model(data)
            teacher_logits = sum([net(data).detach() for net in nets]) / len(select_list)

            loss = divergence(output, teacher_logits)
            loss.backward()
            central_node.optimizer.step()

    return central_node

# FedBE
class SWAG_server(torch.nn.Module):
    def __init__(self, base_model, avg_model=None, max_num_models=25, var_clamp=1e-5, concentrate_num=1):
        self.base_model = base_model
        self.max_num_models=max_num_models
        self.var_clamp=var_clamp
        self.concentrate_num = concentrate_num
        self.avg_model = avg_model
         
    def compute_var(self, mean, sq_mean): 
        var_dict = {}
        for k in mean.keys():
          var = torch.clamp(sq_mean[k] - mean[k] ** 2, self.var_clamp) 
          var_dict[k] = var 

        return var_dict

    def compute_mean_sq(self, teachers):
        w_avg = {}
        w_sq_avg = {}
        w_norm ={}
        
        for k in teachers[0].keys():
            if "batches_tracked" in k: continue
            w_avg[k] = torch.zeros(teachers[0][k].size())
            w_sq_avg[k] = torch.zeros(teachers[0][k].size())
            w_norm[k] = 0.0 
          
        for k in w_avg.keys():
            if "batches_tracked" in k: continue
            for i in range(0, len(teachers)):
              grad = teachers[i][k].cpu()- self.base_model[k].cpu()
              norm = torch.norm(grad, p=2)
              
              grad = grad/norm
              sq_grad = grad**2
              
              w_avg[k] += grad
              w_sq_avg[k] += sq_grad
              w_norm[k] += norm
              
            w_avg[k] = torch.div(w_avg[k], len(teachers))
            w_sq_avg[k] = torch.div(w_sq_avg[k], len(teachers))
            w_norm[k] = torch.div(w_norm[k], len(teachers))
            
        return w_avg, w_sq_avg, w_norm
        
    def construct_models(self, teachers, mean=None, mode="dir"):
      if mode=="gaussian":
        w_avg, w_sq_avg, w_norm= self.compute_mean_sq(teachers)
        w_var = self.compute_var(w_avg, w_sq_avg)      
        
        mean_grad = copy.deepcopy(w_avg)
        for i in range(self.concentrate_num):
          for k in w_avg.keys():
            mean = w_avg[k]
            var = torch.clamp(w_var[k], 1e-6)
            
            eps = torch.randn_like(mean)
            sample_grad = mean + torch.sqrt(var) * eps * 0.1
            mean_grad[k] = (i*mean_grad[k] + sample_grad) / (i+1)
        
        for k in w_avg.keys():
          mean_grad[k] = mean_grad[k]*1.0*w_norm[k] + self.base_model[k].cpu()
          
        return mean_grad  
      
      elif mode=="random":
        num_t = 3
        ts = np.random.choice(teachers, num_t, replace=False)
        mean_grad = {}
        for k in ts[0].keys():
          mean_grad[k] = torch.zeros(ts[0][k].size())
          for i, t in enumerate(ts):
            mean_grad[k]+= t[k]
          
        for k in ts[0].keys():
          mean_grad[k]/=num_t  
          
        return mean_grad
      
      elif mode=="dir":
        proportions = np.random.dirichlet(np.repeat(1.0, len(teachers)))
        mean_grad = {}
        for k in teachers[0].keys():
          mean_grad[k] = torch.zeros(teachers[0][k].size())
          for i, t in enumerate(teachers):
            mean_grad[k]+= t[k]*proportions[i]
          
        for k in teachers[0].keys():
          mean_grad[k]/=sum(proportions)  

        return mean_grad   


def fedbe(args, prev_global_param, central_node, client_nodes, select_list):
    # generate teachers
    nets = []
    base_teachers = []

    fedavg_model = init_model(args.local_model, args).cuda()
    swag_model = init_model(args.local_model, args).cuda()
    fedavg_model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
    nets.append(copy.deepcopy(fedavg_model))

    for client_idx in select_list:
        client_nodes[client_idx].model.cuda().eval()
        nets.append(copy.deepcopy(client_nodes[client_idx].model))
        base_teachers.append(copy.deepcopy(client_nodes[client_idx].model.state_dict()))

    # generate swag model
    swag_server = SWAG_server(prev_global_param, avg_model=copy.deepcopy(central_node.model.state_dict()), concentrate_num=1)
    w_swag = swag_server.construct_models(base_teachers, mode='gaussian') 
    swag_model.load_state_dict(w_swag)
    nets.append(copy.deepcopy(swag_model))  

    # train and update
    central_node.model.cuda().train()
    for _ in range(args.server_epochs):
        train_loader = central_node.validate_set 

        for _, (data, target) in enumerate(train_loader):
            central_node.optimizer.zero_grad()
            # train model
            data, target = data.cuda(), target.cuda()

            output = central_node.model(data)
            teacher_logits = sum([net(data).detach() for net in nets]) / len(nets)

            loss = divergence(output, teacher_logits)
            loss.backward()
            central_node.optimizer.step()

    return central_node

def feddyn(args, central_node, agg_weights, client_nodes, select_list):
    '''
    server function for feddyn
    '''

    # update server's state
    uploaded_models = []
    for i in select_list:
        uploaded_models.append(copy.deepcopy(client_nodes[i].model))

    model_delta = copy.deepcopy(uploaded_models[0])
    for param in model_delta.parameters():
        param.data = torch.zeros_like(param.data)

    for idx, client_model in enumerate(uploaded_models):
        for server_param, client_param, delta_param in zip(central_node.model.parameters(), client_model.parameters(), model_delta.parameters()):
            delta_param.data += (client_param - server_param) * agg_weights[idx]

    for state_param, delta_param in zip(central_node.server_state.parameters(), model_delta.parameters()):
        state_param.data -= args.mu * delta_param

    # aggregation
    central_node.model = copy.deepcopy(uploaded_models[0])
    for param in central_node.model.parameters():
        param.data = torch.zeros_like(param.data)
        
    for idx, client_model in enumerate(uploaded_models):
        for server_param, client_param in zip(central_node.model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * agg_weights[idx]

    for server_param, state_param in zip(central_node.model.parameters(), central_node.server_state.parameters()):
        server_param.data -= (1/args.mu) * state_param

    return central_node

def ccvr(args, central_node, local_protos_list):
    # compute global feature mean and var
    all_mean = []
    all_cov = []
    for idx in range(args.node_num):
        all_mean.append(local_protos_list[idx]['mean'])
        all_cov.append(local_protos_list[idx]['var'])

    global_mean_all = {index: [] for index in range(central_node.num_classes)}
    global_mean_avg = {index: [] for index in range(central_node.num_classes)}

    # compute mean for each class
    for client_num, feature_one in enumerate(all_mean):
        for class_num, feature in feature_one.items():
            global_mean_all[class_num].append(feature)
    # sum
    for index in range(central_node.num_classes):
        global_one_class = global_mean_all[index]
        global_mean_avg[index] = sum(global_one_class) / len(global_one_class)

    # compute covariance
    global_cov_all1 = {index: [] for index in range(central_node.num_classes)}
    global_cov_avg1 = {index: [] for index in range(central_node.num_classes)}
    # compute mean for each class
    for client_num, feature_one in enumerate(all_cov):
        for class_num, feature in feature_one.items():
            global_cov_all1[class_num].append(feature)
    # sum
    for index in range(central_node.num_classes):
        global_one_class = global_cov_all1[index]
        global_cov_avg1[index] = sum(global_one_class) / len(global_one_class)

    # the second part of the covariance
    global_cov_all2 = {index: [] for index in range(central_node.num_classes)}
    global_cov_avg2 = {index: [] for index in range(central_node.num_classes)}
    # compute mean for each class
    for client_num, feature_one in enumerate(all_mean):
        for class_num, feature in feature_one.items():
            global_cov_all2[class_num].append(feature * feature)
        # sum
    for index in range(central_node.num_classes):
        global_one_class = global_cov_all2[index]
        global_cov_avg2[index] = sum(global_one_class) / len(global_one_class)

    global_cov_avg3 = {index: [] for index in range(central_node.num_classes)}
    for index in range(central_node.num_classes):
        global_one_class1 = global_cov_avg1[index]
        global_one_class2 = global_cov_avg2[index]
        temp = global_one_class1 + global_one_class2
        avg_part3 = global_mean_avg[index]
        part3 = avg_part3 * avg_part3
        global_cov_avg3[index] = temp - part3

    # train the classifier
    global_mean = global_mean_avg
    global_cov = global_cov_avg3
    sample_num = [10] * central_node.num_classes

    # print(sum(sample_num))
    # sampling
    sampling_all = []
    label_all = []
    for i in range(central_node.num_classes):
        for _ in range(sample_num[i]):
            generate_sample = torch.normal(global_mean[i], global_cov[i]).cuda()
            sampling_all.append(generate_sample)
            label_one = torch.tensor(i).cuda()
            label_all.append(label_one)

    sampling_all = torch.stack(sampling_all, dim=0).cuda()
    label_all = torch.stack(label_all, dim=0).cuda()

    dst_train_syn_ft = TensorDataset(sampling_all, label_all)

    central_node.model = freeze_layers(central_node.model, central_node.base_key)

    optimizer_ft_net = torch.optim.SGD(central_node.model.linear_head.parameters(), lr=0.01)  # optimizer_img for synthetic data
    
    for epoch in range(100):
        trainloader_ft = torch.utils.data.DataLoader(dataset=dst_train_syn_ft,
                                    batch_size=128,
                                    shuffle=True)
        for data_batch in trainloader_ft:
            images, labels = data_batch
            images, labels = images.cuda(), labels.cuda()
            outputs = central_node.model.linear_head(images)
            loss_net = F.cross_entropy(outputs, labels)
            optimizer_ft_net.zero_grad()
            loss_net.backward()
            optimizer_ft_net.step()

    return central_node