import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt 


def train(args, model, train_loader, optimizer, epoch, writer, scheduler=False, step_flag=False):
    
    model.train()

    loss_fn = nn.GaussianNLLLoss(reduction='mean', eps=1e-3)
    
    total_samples = 0
    total_loss = 0
    batch_interval = 0
    
    start_time = time.time()

    for batch_idx, batch_data in enumerate(train_loader):
        
        batch_interval += 1
        
        total_samples += len(batch_data[0])  
        batch_inputs, batch_targets = batch_data

        optimizer.zero_grad()
        
        means, variances = model(batch_inputs)
        batch_targets = batch_targets.squeeze(-1)
                
        loss = loss_fn(means, batch_targets, variances)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        if step_flag:
            scheduler.step()
            
        if ((batch_idx+1) % args.log_interval == 0) or (batch_idx+1 == len(train_loader)):
        
            avg_loss = total_loss / batch_interval
            elapsed = time.time() - start_time

            print('| Epoch {:3d} | {:5d}/{:5d} batches | {:5d}/{:5d} samples |'
                  ' lr {:2.5f} | ms/batch {:5.2f} | loss {:5.8f} |'.format(
                      epoch, batch_idx+1, len(train_loader), total_samples, len(train_loader.sampler),
                      optimizer.param_groups[0]['lr'], elapsed * 1000 / batch_interval, avg_loss))
            
            batch_iter = (epoch-1) * len(train_loader) + batch_idx+1
            
            if writer is not None:
                writer.add_scalar('Train loss', avg_loss, batch_iter)
                writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], batch_iter)
                
            if args.weight_hist:
                if writer is not None:
                    writer.add_histogram("Dense Layer weight", model.linear.weight.flatten(),
                                         global_step = batch_iter, bins = 'tensorflow')
                    writer.add_histogram("Dense Layer bias", model.linear.bias.flatten(),
                                         global_step = batch_iter, bins = 'tensorflow')
                    block_indx = 1
                    for temp_block in model.tcn.network:
                        layer_indx = 1
                        for layer in temp_block.net:
                            if isinstance(layer, nn.Conv1d):
                                tag = "Temporal Block: "+str(block_indx)+" Convolution layer: "+str(layer_indx)
                                writer.add_histogram(tag+" weight", layer.weight.flatten(),
                                                     global_step = batch_iter, bins = 'tensorflow')
                                writer.add_histogram(tag+" bias", layer.bias.flatten(),
                                                     global_step = batch_iter, bins = 'tensorflow')
                                layer_indx += 1
                        if temp_block.downsample is not None:
                            tag = "Temporal Block: "+str(block_indx)+" Downsample"
                            writer.add_histogram(tag+" weight", temp_block.downsample.weight.flatten(),
                                                 global_step = batch_iter, bins = 'tensorflow')
                            writer.add_histogram(tag+" bias", temp_block.downsample.bias.flatten(),
                                                 global_step = batch_iter, bins = 'tensorflow')
                        block_indx += 1
                
            start_time = time.time()
            total_loss = 0
            batch_interval = 0
            

def evaluate(model, test_loader):

    model.eval()

    loss_fn = nn.GaussianNLLLoss(reduction='none', eps=1e-3)
    
    with torch.no_grad():
                
        for data in test_loader:

            inputs, targets = data
            targets = targets.squeeze(-1)
            
            means, variances = model(inputs)

            loss = loss_fn(means, targets, variances)

            means = means.cpu()
            variances = variances.cpu()
            targets = targets.cpu()
            
            prob_density = (1.0/np.sqrt(2.0*np.pi*variances))*np.exp(-0.5*(means-targets)**2/variances)
            absolute_diff = np.abs(means-targets)
        
        return loss, means, variances, prob_density, absolute_diff


def singular(mean, var, test_loader):

    loss_fn = nn.GaussianNLLLoss(reduction='none', eps=1e-3)
          
    for data in test_loader:

        inputs, targets = data
        
        targets = targets.squeeze(-1)
        means = torch.full([len(targets)], mean)
        variances = torch.full([len(targets)], var)

        targets = targets.cpu()
        means = means.cpu()
        variances = variances.cpu()
        
        loss = loss_fn(means, targets, variances)
        
        prob_density = (1.0/np.sqrt(2.0*np.pi*variances))*np.exp(-0.5*(means-targets)**2/variances)
        absolute_diff = np.abs(means-targets)
        
        return loss, prob_density, absolute_diff


def graph_test(model, test_loader, numplots):

    model.eval()

    numcols = 4
    numrows = int(np.ceil(numplots/numcols))

    fig, axs = plt.subplots(numrows, numcols, figsize=(10,2*numrows))
    
    fig.suptitle('Test Set: Prediction vs Target', fontsize=15)
    fig.text(0.5, 0.04, 'Return', ha='center', va='center')
    fig.text(0.06, 0.5, 'Probability Density', ha='center', va='center', rotation='vertical')
    

    with torch.no_grad():
                
        for data in test_loader:

            inputs, targets = data
            targets = targets.squeeze(-1)
            means, variances = model(inputs)

            means = means.cpu()
            variances = variances.cpu()
            targets = targets.cpu()

        for graph_ind in range(numplots):

            mu = means[graph_ind]
            sigma = math.sqrt(variances[graph_ind])

            max_r = max(abs(mu), abs(targets[graph_ind]), 2.0)
            x = np.linspace(-1.2*max_r, 1.2*max_r, 100)

            ax = axs[int(np.floor((graph_ind)/numcols)),np.mod(graph_ind,numcols)]
            
            ax.plot(x, stats.norm.pdf(x, mu, sigma))
            ax.set_ylim(ymin = 0, ymax = 1.0)
            ax.axvline(x = targets[graph_ind], color='r')
            ax.axvline(x = mu, color='b')
            ax.axvline(x = 0, color='k')
        

def find_lr(args, model, train_loader, optimizer, beta = 0.98):

    num_batches = len(train_loader) - 1
    mult_factor = (args.lr_max / args.lr_init) ** (1/(num_batches*args.epochs))

    lr = args.lr_init
    optimizer.param_groups[0]['lr'] = lr
    loss_fn = nn.GaussianNLLLoss(reduction='mean', eps=1e-3)
    
    avg_loss = 0
    best_loss = 0    
    losses = []
    log_lrs = []

    for epoch in range(0, args.epochs):
    
        for batch_idx, batch_data in enumerate(train_loader):
    
            batch_inputs, batch_targets = batch_data
            optimizer.zero_grad()
            
            means, variances = model(batch_inputs)
            batch_targets = batch_targets.squeeze(-1)
            loss = loss_fn(means, batch_targets, variances)
    
            avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**((epoch*num_batches)+batch_idx+1))
            
            if ((epoch*num_batches)+batch_idx) > 50 and (smoothed_loss > 2 * max(best_loss, losses[-1]) or  math.isnan(smoothed_loss)):
                return log_lrs, losses
                
            if smoothed_loss < best_loss or batch_idx == 0:
                best_loss = smoothed_loss
    
            losses.append(smoothed_loss)
            log_lrs.append(np.log10(lr))
            
            loss.backward()
            optimizer.step()
        
            lr *= mult_factor
            optimizer.param_groups[0]['lr'] = lr
        
    return log_lrs, losses

