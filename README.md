# ComNet

ComNet is a TCN designed to predict a mean and variance for the returns on commodity pricing data based on any number of timeseries input channels,
typically prices or indicators of the commodity itself along with related tickers such as other commodities, Forex, Bond yields etc.

## Inputs

I have provided some raw inputs in the Data folder. These inputs need to be processed before they can be used by the network, this is done in Financial_Data_Processing.
Processing the data produces different feature dataframes, for example for different subsets of tickers or using technical indicators instead of price data.
Different target dataframes are also produced, for example those for different commodities or those to match the technical indicator feature dataframes, in which some initial dates are omitted.
Before the data is saved any price data is converted into a percentage return including the target price. All the Input data is saved under Data/Inputs.

## Network

The network is a TCN largely based on the design of [Bai, Kolter, Koten](https://arxiv.org/abs/1803.01271) with some minor differences
(e.g. padding is added using a CausalConv1d layer instead of a regular Conv1d and a chomp). 
The final layer of the network however is dense and produces two outputs correspoding to a mean and a variance.
A final transformation is applied to the variance after the dense network to limit its raange to positive numbers,
this transformation was chosen such that for large values the transformation becomes linear:

$(x+e^{-x/2})*\sigma(x)$.

## Training

The network was build to use the Apple MPS Framework for GPU acceleration but should also work with CUDA
furthermore the training should work on a distributed network although this functionality has not yet been tested.
Training is done using dataloaders for the train and test sets which are custom datasets that select data between start and end dates specified in the main script.
Hyperparameters are specified on the command line using the argparse function, options are given in the table below.
Before training begins the hyperparameters are saved to a json file; 
at specified intevals during training the model is saved to a pkl file and statistics are both printed and are saved to tensorboard.
