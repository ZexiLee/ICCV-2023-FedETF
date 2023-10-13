import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--iid', type=int, default=0,
                        help='set 1 for iid, and 0 for noniid (dir. sampling)')
    parser.add_argument('--batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--validate_batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, 
                    help="dirichlet_alpha")

    # System
    parser.add_argument('--device', type=str, default='1',
                        help="cuda device: {cuda, cpu}")
    parser.add_argument('--node_num', type=int, default=20, 
                        help="Number of nodes") 
    parser.add_argument('--T', type=int, default=200, 
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=3, 
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="Type of algorithms:{mnist, cifar10,cifar100, fmnist, tinyimagenet}") 
    parser.add_argument('--select_ratio', type=float, default=1.0,
                    help="the ratio of client selection in each round")
    parser.add_argument('--local_model', type=str, default='CNN',
                        help='Type of local model: {CNN, ResNet20, ResNet18}')
    parser.add_argument('--random_seed', type=int, default=10,
                        help="random seed for the whole experiment")
    parser.add_argument('--exp_name', type=str, default='FirstTable',
                        help="experiment name")

    # Server function
    parser.add_argument('--server_method', type=str, default='fedavg',
                        help="FedAvg, or others")
    parser.add_argument('--server_epochs', type=int, default=20,
                        help="optimizer epochs on server for FedDF etc.")
    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help="type of fusion optimizer, adam or sgd")
    parser.add_argument('--server_valid_ratio', type=float, default=0.02, 
                    help="the ratio of proxy set in the central server")  
                        
    # Client function
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="client method")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--client_valid_ratio', type=float, default=0.3,
                    help="the ratio of validate set in the clients for testing personalization")  
    parser.add_argument('--lr', type=float, default=0.04,  
                        help='learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--mu', type=float, default=0.001,
                        help="proximal term mu")

    # For FedETF
    parser.add_argument('--scaling_train', type=float, default=1.0,
                        help="scaling hyperparameter for training")

    args = parser.parse_args()

    return args
