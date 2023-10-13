
from random import random
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
from server_funct import *
import wandb
from client_funct import *

if __name__ == '__main__':

    dataset_list = ['cifar10', 'cifar100']
    alpha_list = [0.1, 0.05]
    random_seed_list = [8, 9, 10]

    for dataset in dataset_list:
        for alpha in alpha_list:
            testacc_list = []
            peracc_list = []
            for random_seed in random_seed_list:
                # try:
                args = args_parser()
                args.dataset = dataset
                args.dirichlet_alpha = alpha
                args.random_seed = random_seed

                # TODO delect
                args.exp_name = 'NCFL'
                args.node_num = 20
                args.iid = 0
                args.noniid_type = 'dirichlet'
                # args.random_seed = 7
                # args.dirichlet_alpha = 0.05
                args.local_model = 'ResNet20' 
                # args.dataset = 'tinyimagenet'
                args.T = 100
                args.E = 3

                if args.dataset in ['cifar100', 'tinyimagenet']:
                    args.lr = 0.01

                root_path = '/code_root'
                output_path = 'results/date'

                setting_name =  args.exp_name + '_' + args.dataset + '_' + args.local_model + '_nodenum' + str(args.node_num) + '_dir' + str(args.dirichlet_alpha) +'_E'+ str(args.E)  + '_C' + str(args.select_ratio) \
                + '_' + args.server_method + '_' + args.client_method + '_seed' + str(args.random_seed)
                initial_model = torch.load(os.path.join(root_path, output_path, setting_name+'_finalmodel.pth'))

                args.client_method = 'ccvr'
                args.server_method = 'ccvr'

                setup_seed(args.random_seed)

                if args.client_method == 'feddyn':
                    args.mu = 0.01
                elif args.client_method in ['fedprox', 'ditto']:
                    args.mu = 0.001
                

                # TODO
                # wandb.init(
                #     config = args,
                #     project = 'NCFL',
                #     name = setting_name , tags = args.exp_name
                # )

                # set GPU device
                # args.device = '1'
                os.environ['CUDA_VISIBLE_DEVICES'] = args.device
                torch.cuda.set_device('cuda:'+args.device)

                data = Data(args)

                sample_size = []
                for i in range(args.node_num): 
                    sample_size.append(len(data.train_loader[i]))
                size_weights = [i/sum(sample_size) for i in sample_size]
                # print('size-based weights',size_weights)

                # initialize the central node
                # num_id equals to -1 stands for central node
                central_node = Node(-1, data.test_loader[0], data.test_set, args)


                central_node.model.load_state_dict(initial_model)

                # initialize the client nodes
                client_nodes = {}
                for i in range(args.node_num): 
                    client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args) 
                    client_nodes[i].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))

                test_acc_recorder = []

                if args.select_ratio == 1.0:
                    select_list_recorder = [[i for i in range(args.node_num)] for _ in range(args.T)]
                else:
                    select_list_recorder = torch.load(os.path.join(root_path, 'outputs/0915/','num'+ str(args.node_num)+'_ratio'+str(args.select_ratio)+ '_select_list_recorder.pth'))

                # print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
                print(setting_name)
                # Client selection
                select_list = select_list_recorder[0]
                acc = validate(args, central_node, which_dataset = 'local')
                print('before ccvr, global model test acc is ', acc)
                avg_client_acc = Client_validate(args, client_nodes, select_list)
                print('before ccvr, personalization acc is ', avg_client_acc)

                # ccvr process
                # Local update
                client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)
                # Server aggregation
                central_node = Server_update(args, central_node, client_nodes, select_list, size_weights)
                acc = validate(args, central_node, which_dataset = 'local')
                print('after ccvr, global model test acc is ', acc)


                # Local finetuning: personalization process
                args.client_method = 'local_train'
                select_list = [idx for idx in range(len(client_nodes))]
                client_nodes, train_loss = Client_personalization(args, client_nodes, central_node, select_list)
                avg_client_acc = Client_validate(args, client_nodes, select_list)
                print('after ccvr, personalization acc is ', avg_client_acc)

                testacc_list.append(acc)
                peracc_list.append(avg_client_acc)
                # except:
                #     pass
            print('--------------------------')
            print(setting_name)
            print('all, testacc is', sum(testacc_list)/len(testacc_list))
            print('all, peracc is', sum(peracc_list)/len(peracc_list))
            print('all, testacc std is', np.std(testacc_list))
            print('all, peracc std is', np.std(peracc_list))
                