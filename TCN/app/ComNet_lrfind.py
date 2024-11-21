
import argparse
import sys, time, os, shutil, json, functools

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './app/src')

from Dataset import *
from TCN import *
from Train_Eval import *

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
RANK = int(os.environ.get("RANK",0))


def parse_arguments(argv):
    
    parser = argparse.ArgumentParser(description='args.')

    parser.add_argument('--inputs_file_name', default='../../Data/Input/Basic/input_data.pkl',
                        help='directory where input data is stored')
    parser.add_argument('--targets_file_name', default='../../Data/Input/Basic/target_data.pkl',
                        help='directory where target data is stored')
    parser.add_argument('--save_dir', default='./models',
                        help='directory where the models and logs are to be saved')
    parser.add_argument('--name',required=True ,
                        help='name of the model')
    parser.add_argument('--version',default='latest',
                        help='version of the model')
    parser.add_argument('--gpu_accel', action='store_true', default=True,
                        help='whether or not to use GPU acceleration')
    parser.add_argument('--model_config',type=json.loads, required=True,
                        help='json string configuration of the model')
    parser.add_argument('--optimizer', choices=['AdamW','SGD','SGD_nesterov'], default='SGD',
                        help='Optimizer choice')
    parser.add_argument('--lr_init', type=float, default='1e-5',
                        help='initial learning rate')
    parser.add_argument('--lr_max', type=float, default='1e-1',
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default='0.001',
                        help='weight decay') 
    parser.add_argument('--batch_size', type=int, default='16',
                        help='input batch size for training')   
    parser.add_argument('--beta', type=float, default='0.98',
                        help='beta for EMA smoothing of LR-Loss curve') 
    parser.add_argument('--epochs', type=int, default='10',
                        help='number of epochs to train')
    parser.add_argument('--num_runs', type=int, default='5',
                        help='number of runs to average over')
    
    args = parser.parse_args()
    
    return args
    

def init_logs(args,save_path):
    
    logs_path=os.path.join(save_path,'logs')
    
    logs_path=os.path.join(logs_path,'lr_find')
        
    if RANK==0:
        if os.path.exists(save_path):
            shutil.rmtree(save_path,ignore_errors=True)
            os.makedirs(save_path)
        writer = SummaryWriter(logs_path)
    else:
        writer =None
        
    return writer
    

def parse_config(args, save_path):
    
    model_name = '.'.join([args.name,args.version])
    config_path = os.path.join(save_path, model_name+'.json')

    config = args.model_config
    config['total_epochs'] = args.epochs
    
    job_args = dict(vars(args))
    del job_args['model_config']
    config['job_args'] = job_args    
    
    if RANK==0:
        with open(config_path ,'w') as file:
            json.dump(config, file)
            
    return config, model_name 
    

def store_lr(args, model_name):
    
    lr_path = os.path.join(save_path, model_name+'_lr.json')  
    
    if RANK==0:
        with open(config_path ,'w') as file:
            json.dump(config, file)
    

def dataset_loaders(args, train_dataset, test_dataset, device, epoch):
            
    if dist.is_initialized():
        train_sampler = data.distributed.DistributedSampler(train_dataset)
        test_sampler = data.distributed.DistributedSampler(test_dataset)
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
    else:
        train_sampler=None
        test_sampler=None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, sampler=train_sampler,
                              collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, sampler=test_sampler,
                             collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    
    return train_loader, test_loader 
        

def main(args):

    # GPU Acceleration
    
    if args.gpu_accel and torch.backends.mps.is_available():
        device_flag = 'mps'
        device = torch.device("mps")
        print("Using Apple MPS Framework for GPU acceleration with {} local device/s".format(torch.mps.device_count()),"\n")
    elif args.gpu_accel and torch.cuda.is_available():
        device_flag = 'cuda'
        device = torch.device("cuda")
        print("Using Nvidia CUDA for GPU acceleration with {} local device/s".format(torch.cuda.device_count()),"\n")
    else:
        device_flag = 'cpu'
        print("Not using GPU acceleration","\n")

    # Use PyTorch distributed if there are multiple GPUs
    
    if dist.is_available() and WORLD_SIZE > 1:        
        print('Using distributed PyTorch with {} backend. \tNode rank: {}/{}'.format(args.backend,RANK+1,WORLD_SIZE),"\n")
        if device_flag == 'cpu':
            dist.init_process_group(backend='gloo')
        else:
            dist.init_process_group(backend='nccl')

    # Create save paths and writer object for outputs

    save_path = os.path.join(args.save_dir, args.name, args.version)
    writer = init_logs(args, save_path)
    config, model_name = parse_config(args, save_path)
    model_args = config['model_args']

    # Create a Dataset objects for inputs and target
    
    Train_Dataset = Dataset(args.inputs_file_name, args.targets_file_name,
                            startdate='2005-01-01',enddate='2019-01-01', history_length=model_args["history_length"])
    Test_Dataset = Dataset(args.inputs_file_name, args.targets_file_name,
                           startdate='2019-01-01',enddate='2022-01-01', history_length=model_args["history_length"])

    list_lrs = []
    list_losses = []
    
    fig = plt.figure() 
    plt.title("Individual Runs")
    plt.xlabel("Log lr")
    plt.ylabel("Loss")

    print("LR finder begins...","\n")
    
    for run_num in range(args.num_runs):

        print("Run number: ", run_num + 1)
    
        # Create network model
    
        TCN_model = TCN(**model_args).set_device(device)
            
        # Create loader objects
        
        Train_Loader, Test_Loader = dataset_loaders(args, Train_Dataset, Test_Dataset, device, 0)
            
        # Print number of devices if training is distributed
        
        if dist.is_initialized():
            
            if device_flag == 'mps':
                model = nn.parallel.DistributedDataParallel(model)
            elif device_flag == 'cuda':
                model = nn.parallel.DistributedDataParallel(model)
            else:
                local_rank = int(os.environ["LOCAL_RANK"])
                model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)
        
        # Set up optimizer (only use SGD for lr_find)
        
        optim_map = {'AdamW' : optim.AdamW,
                     'SGD': functools.partial(optim.SGD, momentum = 0.9),
                     'SGD_nesterov': functools.partial(optim.SGD, momentum = 0.9, nesterov = True)}
        
        optimizer = optim_map[args.optimizer](TCN_model.parameters(),
                                              lr = args.lr_init,
                                              weight_decay = args.weight_decay,
                                              fused = True)
        
        # Begin lr_find cycle
    
        log_lrs, losses = find_lr(args, TCN_model, Train_Loader, optimizer, beta = args.beta)
    
        plt.plot(log_lrs[20:-20], losses[20:-20])
        print("length of loss array in this run: ", len(losses))
        print("lr with the minimum loss for this run: ",log_lrs[np.argmin(losses[5:-5])])
        print()

        list_lrs.append(log_lrs)
        list_losses.append(losses)

    # Write individual runs graph to file

    writer.add_figure('Individual Runs', fig)

    # Plot aggregate loss vs lr
    
    max_len = len(min(list_lrs, key=len))

    for log_lr in list_lrs:
        del log_lr[max_len:]
    for losses in list_losses:
        del losses[max_len:]

    array_lrs = np.array(list_lrs)
    array_losses = np.array(list_losses)
    avg_lrs = np.average(array_lrs, axis=0)
    avg_losses = np.average(array_losses, axis=0)

    lr_max = avg_lrs[np.argmin(avg_losses)]

    fig = plt.figure()
    plt.plot(avg_lrs, avg_losses)
    plt.title("Aggregate")
    plt.xlabel("Log lr")
    plt.ylabel("Loss")

    print()
    print("Overall result: ")
    print("Log lr with the minimum loss in aggregate: ",lr_max)
    print("This equates to a learning rate of: {:1.3e}".format(10**lr_max))
    print()

    writer.add_figure('Aggregate Result', fig)

    # Save lr statistics

    lr_path = os.path.join(save_path, model_name+'_lr.json')  
    lr_stats = {"log_lr_max": lr_max, "lr_max": 10**lr_max}
    if RANK==0:
        with open(lr_path ,'w') as file:
            json.dump(lr_stats, file)
    
    writer.close()
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))