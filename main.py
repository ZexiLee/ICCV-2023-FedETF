
import time
import torch
from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
import numpy as np
import os
import torch.nn as nn
import copy
import torch.optim as optim
import torch.nn.functional as F
import math
from pyhessian import hessian
from server_funct import *
import wandb
from client_funct import *

if __name__ == '__main__':

    args = args_parser()

    ##### Exp settings #####
    ##### change it for different exps #####
    args.client_method = 'fedetf'
    args.exp_name = 'NCFL'
    args.node_num = 20
    args.iid = 0
    args.random_seed = 10
    args.dirichlet_alpha = 0.05
    args.local_model = 'ResNet20' 
    args.dataset = 'cifar10'
    args.T = 200
    args.E = 3

    # set the server method automatically
    args = set_server_method(args)
    # set the random seed for controlling the randomness
    setup_seed(args.random_seed)


    # setting hyperparameter for feddyn, fedprox, and ditto
    if args.client_method == 'feddyn':
        args.mu = 0.01
    elif args.client_method in ['fedprox', 'ditto']:
        args.mu = 0.001
    
    # hyperparameter for different datasets
    if args.dataset in ['cifar100', 'tinyimagenet']:
        args.lr = 0.01
    
    setting_name =  args.exp_name + '_' + args.dataset + '_' + args.local_model + '_nodenum' + str(args.node_num) + '_dir' + str(args.dirichlet_alpha) +'_E'+ str(args.E)  + '_C' + str(args.select_ratio) \
    + '_' + args.server_method + '_' + args.client_method + '_seed' + str(args.random_seed)

    root_path = '/code_root'
    output_path = 'results/date'

    wandb.init(
        config = args,
        project = 'NCFL',
        name = setting_name , tags = args.exp_name
    )

    # for ccvr
    if args.client_method == 'ccvr':
        args.client_method = 'local_train'
        args.server_method = 'fedavg'
        method_cache = 'ccvr'
    else:
        method_cache = None

    # set GPU device
    # args.device = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # torch.cuda.set_device('cuda:'+args.device)

    data = Data(args)

    sample_size = []
    for i in range(args.node_num): 
        sample_size.append(len(data.train_loader[i]))
    size_weights = [i/sum(sample_size) for i in sample_size]
    print('size-based weights',size_weights)

    # initialize the central node
    # num_id equals to -1 stands for central node
    central_node = Node(-1, data.test_loader[0], data.test_set, args)


    # initialize the client nodes
    client_nodes = {}
    for i in range(args.node_num): 
        client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args) 
        client_nodes[i].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))

    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []

    if args.select_ratio == 1.0:
        select_list_recorder = [[i for i in range(args.node_num)] for _ in range(args.T)]
    else:
        select_list_recorder = torch.load(os.path.join(root_path, 'outputs/0915/','num'+ str(args.node_num)+'_ratio'+str(args.select_ratio)+ '_select_list_recorder.pth'))

    for rounds in range(args.T):
        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        print(setting_name)
        # lr_scheduler(rounds, client_nodes, args)

        # for ccvr, last round calibration
        if (rounds == args.T - 1) and method_cache != None:
            args.server_method = method_cache
            args.client_method = method_cache

        # Client selection
        select_list = select_list_recorder[rounds]

        # Local update
        client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)
        avg_client_acc = Client_validate(args, client_nodes, select_list)
        print(args.server_method + args.client_method + ', averaged clients acc is ', avg_client_acc)
        
        # Server aggregation
        central_node = Server_update(args, central_node, client_nodes, select_list, size_weights)
        acc = validate(args, central_node, which_dataset = 'local')
        print(args.server_method + args.client_method + ', global model test acc is ', acc)

        # Recorder
        test_acc_recorder.append(acc)
        try:
            wandb.log({'trainloss': train_loss}, step = rounds)
            wandb.log({'testacc': acc}, step = rounds)
            wandb.log({'peracc': avg_client_acc}, step = rounds)
        except:
            pass
        # record the acc of the last 5 rounds for final acc
        if rounds >= args.T - 5:
            final_test_acc_recorder.update(acc)
        # for the last round, save the model
        if rounds == args.T - 1:
            torch.save(central_node.model.state_dict(), os.path.join(root_path, output_path, setting_name+'_finalmodel.pth'))
            if 'fedetf' in args.client_method :
                torch.save(central_node.model.proto_classifier.proto, os.path.join(root_path, output_path, setting_name+'_etfproto.pth'))

    print('final_testacc', final_test_acc_recorder.value())

    # Local finetuning: personalization process
    select_list = [idx for idx in range(len(client_nodes))]
    client_nodes, train_loss = Client_personalization(args, client_nodes, central_node, select_list)
    avg_client_acc = Client_validate(args, client_nodes, select_list)
    print(args.server_method + args.client_method + ', personalization acc is ', avg_client_acc)

    # save the final result
    final_results = {'final_testacc': final_test_acc_recorder.value(), 'final_peracc': avg_client_acc}
    try:
        wandb.log(final_results)
    except:
        pass
    torch.save(final_results, os.path.join(root_path, output_path, setting_name+'_finalresults.pth'))
    torch.save(test_acc_recorder, os.path.join(root_path, output_path, setting_name+'_recorder.pth'))