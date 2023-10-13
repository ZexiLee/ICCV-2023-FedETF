from cProfile import label
import numpy as np
import torch
import torch.nn.functional as F
from utils import validate, model_parameter_vector, freeze_layers, set_params
import copy
from nodes import Node

##############################################################################
# General client function 
##############################################################################

def receive_server_model(args, client_nodes, central_node):

    for idx in range(len(client_nodes)):
        # models
        if args.client_method == 'fedrep':
            client_nodes[idx].model = set_params(client_nodes[idx].model, copy.deepcopy(central_node.model.state_dict()), client_nodes[idx].head_key)
        elif args.client_method != 'fedproto':
            client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
        
        # protos
        if 'fedetf' in args.client_method:
            client_nodes[idx].model.proto_classifier.load_proto(central_node.model.proto_classifier.proto)
        elif args.client_method == 'fednh':
            client_nodes[idx].prototype.data = copy.deepcopy(central_node.prototype.data)
        elif args.client_method == 'fedproto':
            client_nodes[idx].prototype = copy.deepcopy(central_node.prototype)

    return client_nodes

def Client_update(args, client_nodes, central_node, select_list):
    '''
    client update functions
    '''
    # clients receive the server model 
    client_nodes = receive_server_model(args, client_nodes, central_node)

    # update the global model
    if args.client_method == 'local_train':
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif 'fedetf' in args.client_method:
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedetf(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'fedprox':
        global_model_param = copy.deepcopy(list(central_node.model.parameters()))
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedprox(global_model_param, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'ditto':
        global_model_param = copy.deepcopy(list(central_node.model.parameters()))
        client_losses = []
        for i in select_list:
            # peronalized training
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedprox(global_model_param, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
            # global model training
            for epoch in range(args.E):
                _ = client_localTrain(args, client_nodes[i])

    elif args.client_method == 'feddyn':
        global_model_vector = copy.deepcopy(model_parameter_vector(args, central_node.model).detach().clone())
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_feddyn(global_model_vector, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

            # update old grad
            v1 = model_parameter_vector(args, client_nodes[i].model).detach()
            client_nodes[i].old_grad = client_nodes[i].old_grad - args.mu * (v1 - global_model_vector)

    elif args.client_method == 'fedrod':
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedrod(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'fednh':
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fednh(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
            client_fednh_compute_proto(args, client_nodes[i])

    elif args.client_method == 'fedproto':
        client_losses = []
        for i in select_list:
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedproto(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
            client_fednh_compute_proto(args, client_nodes[i])

    elif 'fedrep' in args.client_method:
        client_losses = []
        for i in select_list:
            # train head
            client_nodes[i].model = freeze_layers(client_nodes[i].model, client_nodes[i].base_key)
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)
            # train base
            client_nodes[i].model = freeze_layers(client_nodes[i].model, client_nodes[i].head_key)
            for epoch in range(args.E):
                _ = client_localTrain(args, client_nodes[i])

    elif args.client_method == 'ccvr':
        client_losses = []
        for i in select_list:
            # epoch_losses = []
            # for epoch in range(args.E):
            #     loss = client_localTrain(args, client_nodes[i])
            #     epoch_losses.append(loss)
            # client_losses.append(sum(epoch_losses)/len(epoch_losses))
            # train_loss = sum(client_losses)/len(client_losses)
            train_loss = 0.0
            client_ccvr_compute_feature_meanvar(args, client_nodes[i])

    else:
        raise ValueError('Undefined client method...')

    return client_nodes, train_loss




def Client_personalization(args, client_nodes, central_node, select_list):

    # finetune the global model on the local datasets
    client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)

    # fedetf: finetune the proto and the projection layer interchangably
    if 'fedetf' in args.client_method:
        for _ in range(20):
            # finetuen the proto
            for i in range(len(client_nodes)):
                for name, param in client_nodes[i].model.named_parameters():
                    param.requires_grad = False
                client_nodes[i].model.proto_classifier.proto.requires_grad = True
                client_nodes[i].optimizer =  torch.optim.SGD([client_nodes[i].model.proto_classifier.proto], lr=0.1)
                for _ in range(3):
                    _ = client_fedetf(args, client_nodes[i], opt = 'celoss')

            # finetuen the projection layer
            for i in range(len(client_nodes)):
                for name, param in client_nodes[i].model.named_parameters():
                    if 'linear_proto' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                client_nodes[i].model.proto_classifier.proto.requires_grad = False
                client_nodes[i].optimizer =  torch.optim.SGD(filter(lambda p: p.requires_grad, client_nodes[i].model.parameters()), 0.05, momentum=args.momentum, weight_decay=args.local_wd_rate)
                for _ in range(3):
                    _ = client_fedetf(args, client_nodes[i], opt = 'celoss')

    return client_nodes, train_loss



def Client_validate(args, client_nodes, select_list):
    '''
    client validation functions, for testing local personalization
    '''
    client_acc = []
    for idx in select_list:
        acc = validate(args, client_nodes[idx])
        # print('client ', idx, ', after  training, acc is', acc)
        client_acc.append(acc)
    avg_client_acc = sum(client_acc) / len(client_acc)

    return avg_client_acc

def client_localTrain(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        
        data, target = data.cuda(), target.cuda()
        _, output_local, _ = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)

# FedETF
def balanced_softmax_loss(logits, labels, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def client_fedetf(args, node, opt = 'balancedloss', loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        
        data, target = data.cuda(), target.cuda()
        feature, _, _ = node.model(data)

        output_local = torch.matmul(feature, node.model.proto_classifier.proto)
        output_local = node.model.scaling_train * output_local

        if opt == 'balancedloss':
            loss_local = balanced_softmax_loss(output_local, target, node.sample_per_class)
        elif opt == 'celoss':
            # For local personalization
            loss_local =  F.cross_entropy(output_local, target)

        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)

def client_fedrod(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data

    # initialize the optimizer of p_head
    p_head_optimizer = torch.optim.SGD(node.p_head.parameters(), lr=args.lr)

    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        p_head_optimizer.zero_grad()

        # train model
        
        data, target = data.cuda(), target.cuda()
        _, logit_g, feature = node.model(data)

        # balanced loss for base and g_head
        loss_local = balanced_softmax_loss(logit_g, target, node.sample_per_class)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

        # ce loss for p_head
        logit_p = node.p_head(feature.detach())
        logit = logit_g.detach() + logit_p
        loss_p =  F.cross_entropy(logit, target)
        loss_p.backward()
        p_head_optimizer.step()

    return loss/len(train_loader)

# fednh
def client_fednh(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data

    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()

        # train model
        data, target = data.cuda(), target.cuda()
        _, _, feature = node.model(data)

        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature, feature_norm)
        normalized_prototype = node.prototype
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        # logits = node.scaling * logits
        logits = node.model.scaling_train * logits

        loss_local =  F.cross_entropy(logits, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)

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


#  fedproto
def client_fedproto(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data
    mse = torch.nn.MSELoss()

    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()

        # train model
        
        data, target = data.cuda(), target.cuda()
        _, logits, feature = node.model(data)

        # ce loss
        loss_local =  F.cross_entropy(logits, target)
        # proto regularization loss
        if node.prototype != None:
            place_hldr = torch.zeros_like(feature)
            for i, yy in enumerate(target):
                y_c = yy.item()
                place_hldr[i, :] = node.prototype[y_c]
            loss_local += 0.1 * mse(place_hldr, feature)

        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)

def client_fedprox(global_model_param, args, node, loss = 0.0):
    if args.client_method == 'fedprox':
        model = node.model
        optimizer = node.optimizer
    elif args.client_method == 'ditto':
        model = node.p_model
        optimizer = node.p_optimizer

    model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        optimizer.zero_grad()
        # train model
        
        data, target = data.cuda(), target.cuda()
        _, output_local, _ = model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        # fedprox update
        optimizer.step(global_model_param)

    return loss/len(train_loader)

def client_feddyn(global_model_vector, args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        
        data, target = data.cuda(), target.cuda()
        _, output_local, _ = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss = loss + loss_local.item()

        # feddyn update
        v1 = model_parameter_vector(args, node.model)
        loss_local += args.mu/2 * torch.norm(v1 - global_model_vector, 2)
        loss_local -= torch.dot(v1, node.old_grad)

        loss_local.backward()
        node.optimizer.step()

    return loss/len(train_loader)


def client_ccvr_compute_feature_meanvar(args, node):
    node.model.train()

    all_class_features = {index: [] for index in range(node.num_classes)}
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        with torch.no_grad():
            
            data = data.cuda()
            _, _, feature = node.model(data)
            for sample_idx, label in enumerate(target):
                all_class_features[int(label)].append(feature[sample_idx])
    all_class_features = {key:val for key, val in all_class_features.items() if val != []}

    all_class_mean_feature = {index: [] for index in all_class_features.keys()}
    all_class_bias_feature = {index: [] for index in all_class_features.keys()}

    for label in all_class_features.keys():
        feature = torch.tensor([np.array(i.cpu().numpy()) for i in all_class_features[label]])
        if not torch.isnan(feature.std(dim=0)).any():
            all_class_bias_feature[label] = feature.std(dim=0)
            all_class_mean_feature[label] = torch.sum(feature, dim=0) / feature.shape[0]

    all_class_bias_feature = {key:val for key, val in all_class_bias_feature.items() if val != []}
    all_class_mean_feature = {key:val for key, val in all_class_mean_feature.items() if val != []}

    node.feature_meanvar = {'mean':all_class_mean_feature, 'var':all_class_bias_feature}

    return
