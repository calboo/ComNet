
import argparse
import sys, time, os, shutil, json, functools

import numpy as np
import multiprocessing as mp

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
    parser.add_argument('--lr_config',type=json.loads,
                        help='json string configuration of the lr scheduler')
    parser.add_argument('--optimizer', choices=['AdamW','SGD','SGD_nesterov'], default='AdamW',
                        help='Optimizer choice')
    parser.add_argument('--lr', type=float, default='5e-4',
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default='0.001',
                        help='weight decay') 
    parser.add_argument('--sample_noise', type=float, default='0.0',
                        help='sample noise') 
    parser.add_argument('--batch_size', type=int, default='16',
                        help='input batch size for training')   
    parser.add_argument('--log_interval', type=int, default='32',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epochs', type=int, default='10',
                        help='number of epochs to train')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='how many epochs to wait before logging eval status')
    parser.add_argument('--weight_hist', action='store_true', default=False,
                        help='whether to produce histograms for layer parameters')
    parser.add_argument('--print_singular', action='store_true', default=False,
                        help='whether to produce evaluation on a singular gaussian based on model output')
    parser.add_argument('--early_stopping', type=int, default='0',
                        help='save model at each eval and stop if test loss spikes, in resume mode load from specified epoch')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume traning of a model from saved version')
 
    args = parser.parse_args()
    
    return args
    

def init_logs(args,save_path):
    
    logs_path=os.path.join(save_path,'logs')

    if args.log_interval == 0:
        args.resume = True

    if args.log_interval>0:
        logs_path=os.path.join(logs_path,'train')
    else:
        logs_path=os.path.join(logs_path,'eval')
        
    if RANK==0:
        if os.path.exists(save_path) and not args.resume:
            shutil.rmtree(save_path,ignore_errors=True)
            os.makedirs(save_path)
        writer = SummaryWriter(logs_path)
    else:
        writer =None
        
    return writer
    

def parse_config(args, save_path):
    
    model_name = '.'.join([args.name,args.version])
    config_path = os.path.join(save_path, model_name+'.json')
    
    if args.resume:
        with open(config_path, 'r') as file:
            config = json.load(file)  
        if args.log_interval>0:
            config['total_epochs'] = config['total_epochs'] + args.epochs
    else:
        config = args.model_config
        config['total_epochs'] = args.epochs
    
    job_args = dict(vars(args))
    del job_args['model_config']
    config['job_args'] = job_args    
    
    if RANK==0:
        with open(config_path ,'w') as file:
            json.dump(config, file)
            
    return config, model_name 
    

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

    # Create network model

    config, model_name = parse_config(args, save_path)
    model_args = config['model_args']
    TCN_model = TCN(**model_args).set_device(device)
        
    # Create a Dataset objects for inputs and target
    
    Train_Dataset = Dataset(args.inputs_file_name, args.targets_file_name,
                            startdate='2005-01-01',enddate='2019-01-01',
                            history_length=model_args["history_length"],
                            sample_noise = args.sample_noise)
    
    Test_Dataset = Dataset(args.inputs_file_name, args.targets_file_name,
                           startdate='2019-01-01',enddate='2022-01-01',
                           history_length=model_args["history_length"])
    
    # Create loader objects
    
    Train_Loader, Test_Loader = dataset_loaders(args, Train_Dataset, Test_Dataset, device, 0)

    # Save network model to tensorboard
    
    Data_iteration = iter(Train_Loader)
    view_input, view_target = next(Data_iteration)
    writer.add_graph(TCN_model,view_input)
    
    # Print model parameters and device memory
    
    if device_flag == 'mps':
        device_mem = torch.mps.driver_allocated_memory()/(1024**2)
        print('Model initialised with {} occupying {:.2f} MB'.format(TCN_model.count_parameters(),device_mem),"\n")        
    elif device_flag == 'cuda':
        device_mem = torch.cuda.memory_allocated()/(1024**2)
        print('Model initialised with {} occupying {:.2f} MB'.format(TCN_model.count_parameters(),device_mem),"\n")
    else:
        print('Model initialised with {}'.format(TCN_model.count_parameters()),"\n")
        
    # Print number of devices if training is distributed
    
    if dist.is_initialized():
        
        if device_flag == 'mps':
            model = nn.parallel.DistributedDataParallel(model)
            print('Model is distributed over {} devices'.format(WORLD_SIZE*torch.mps.device_count()),"\n")  
        elif device_flag == 'cuda':
            model = nn.parallel.DistributedDataParallel(model)
            print('Model is distributed over {} devices'.format(WORLD_SIZE*torch.cuda.device_count()),"\n")
        else:
            local_rank = int(os.environ["LOCAL_RANK"])
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)
            print('Model is distributed over {} devices'.format(WORLD_SIZE),"\n")
    
    # Set up optimizer
    
    optim_map = {'AdamW' : optim.AdamW,
                 'SGD': functools.partial(optim.SGD, momentum = 0.9),
                 'SGD_nesterov': functools.partial(optim.SGD, momentum = 0.9, nesterov = True)}
    
    optimizer = optim_map[args.optimizer](TCN_model.parameters(),
                                          lr = args.lr,
                                          weight_decay = args.weight_decay,
                                          fused = True)

    # Load model and optimizer if resuming training

    path_model = os.path.join(save_path, args.name)
    path_optimizer = os.path.join(save_path, args.name +'_optimizer.pkl')

    if args.resume:
        if args.early_stopping:
            TCN_model.load_state_dict(torch.load(path_model+"_"+str(args.early_stopping)+'.pkl', weights_only=True, map_location=device))
            optimizer.load_state_dict(torch.load(path_optimizer, weights_only=True, map_location=device))
            print('Resumed model: {}'.format(model_name))
        else:
            TCN_model.load_state_dict(torch.load(path_model+'.pkl', weights_only=True, map_location=device))
            optimizer.load_state_dict(torch.load(path_optimizer, weights_only=True, map_location=device))
            print('Resumed model: {}'.format(model_name))

    # Setup scheduler if used

    one_cycle_flag = False

    if args.lr_config:
        
        lr_config = args.lr_config
        sched_args = lr_config['sched_args']

        print("Using scheduler with settings \n", sched_args, "\n")

        if sched_args["method"] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             factor = eval(sched_args["factor"]),
                                                             patience = sched_args["patience"],
                                                             threshold = sched_args["threshold"])

        if sched_args["method"] == 'cosine_warm':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                       T_0 = sched_args["t_0"],
                                                                       T_mult = sched_args["t_mult"])

        if sched_args["method"] == 'one_cycle':
            one_cycle_flag = True
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr = sched_args["max_lr"],
                                                            three_phase = sched_args["three_phase"],
                                                            steps_per_epoch = len(Train_Loader),
                                                            epochs = args.epochs)
            
    # Begin training/eval cycle

    if args.log_interval>0:
        print("Training begins...","\n")
    else:
        config['total_epochs'] = config['total_epochs'] + args.epochs
        print("Evaluation begins...","\n")

    min_test_loss = np.inf
    
    for epoch in range(config['total_epochs']-args.epochs+1, config['total_epochs']+1):

        # Refresh Dataloaders

        Train_Loader, Test_Loader = dataset_loaders(args, Train_Dataset, Test_Dataset, device, epoch)
        
        # Train model

        if args.log_interval>0:
            if args.lr_config:
                train(args, TCN_model, Train_Loader, optimizer, epoch, writer,
                      scheduler = scheduler, step_flag = one_cycle_flag)
            else:
                train(args, TCN_model, Train_Loader, optimizer, epoch, writer)
        
        # Evaluate model
        
        if (epoch % args.eval_interval == 0) or (epoch == config['total_epochs']) or (args.log_interval==0):
            
            loss, means, variances, prob_densities, abs_diffs = evaluate(TCN_model, Test_Loader)

            print("\n","Test statistics","\n")

            print('| Average Loss {:2.5f} | Average Probability Density {:2.5f} | Average MAE on mean {:2.5f} |\n'
                  '| Average mean {:2.5f} | Average absolute value of mean {:2.5f} | Average variance {:2.5f} |\n'.format(
                      torch.mean(loss), torch.mean(prob_densities), torch.mean(abs_diffs),
                      torch.mean(means), torch.mean(abs(means)), torch.mean(variances)))
            
            if args.print_singular:

                # Print evaluation metrics using a singular gaussian based on model predictions

                singular_loss, singular_prob_den, singular_abs_diff = singular(torch.mean(means), torch.mean(variances), Test_Loader)
    
                print('| Singular Loss {:2.5f} | Singular Probability Density {:2.5f} | Singular MAE on mean {:2.5f} |\n'.format(
                          torch.mean(singular_loss), torch.mean(singular_prob_den), torch.mean(singular_abs_diff)))

            # Make a graphs of samples in the test set
            
            if args.log_interval==0:
                graph_test(TCN_model, Test_Loader, numplots = 20)
                
            # Break if test loss spikes

            if args.early_stopping:
                
                min_test_loss = min(min_test_loss, torch.mean(loss))
                if torch.mean(loss) > 1.5 * min_test_loss:
                    print("Test loss is spiking, ending run...","\n")
                    break

            # Write eval metrics to tensorboard event
            
            writer.add_scalar('Average test loss', torch.mean(loss), epoch)
            writer.add_scalar('Average probability density', torch.mean(prob_densities), epoch)
            writer.add_scalar('Average MAE on mean', torch.mean(abs_diffs), epoch)
            writer.add_scalar('Average mean', torch.mean(means), epoch)
            writer.add_scalar('Average absolute value of mean', torch.mean(abs(means)), epoch)
            writer.add_scalar('Average variance', torch.mean(variances), epoch)

            writer.add_histogram("Histogram Probability Density", prob_densities,
                                 global_step = epoch, bins = 'tensorflow')
            writer.add_histogram("Histogram variance", variances,
                                 global_step = epoch, bins = 'tensorflow')
            writer.add_histogram("Histogram Test Loss", torch.clamp(loss, max = torch.quantile(loss, 0.99)),
                                 global_step = epoch, bins = 'tensorflow')
            writer.add_histogram("Histogram MAE on mean", torch.clamp(abs_diffs, max = torch.quantile(abs_diffs, 0.99)),
                                 global_step = epoch, bins = 'tensorflow')

            # Save model and optimizer

            if args.log_interval>0:

                if args.early_stopping:
                    if dist.is_initialized():
                        torch.save(TCN_model.module.state_dict(),path_model+"_"+str(epoch)+'.pkl')
                    else:
                        torch.save(TCN_model.state_dict(),path_model+"_"+str(epoch)+'.pkl')
                    torch.save(optimizer.state_dict(),path_optimizer)
                    print('Model saved to path: {}\n'.format(path_model+"_"+str(epoch)+'.pkl'))
                else:
                    if dist.is_initialized():
                        torch.save(TCN_model.module.state_dict(),path_model+'.pkl')
                    else:
                        torch.save(TCN_model.state_dict(),path_model+'.pkl')
                    torch.save(optimizer.state_dict(),path_optimizer)
                    print('Model saved to path: {}\n'.format(path_model+'.pkl'))

        # lr scheduler step

        if args.lr_config:
            if sched_args["method"] == 'plateau': 
                scheduler.step(torch.mean(loss))
            elif sched_args["method"] == 'cosine_warm':
                scheduler.step()

    writer.close()
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
