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

$y = (x+e^{-x/2})*\sigma(x)$,

where $\sigma$ is the sigmoid function and $y$ is the output variance.

## Training

Training is called notebook Main Script Experiments, which calls the main script in ComNet_Train.py.
The network was build to use the Apple MPS Framework for GPU acceleration but should also work with CUDA
furthermore the training should work on a distributed network although this functionality has not yet been tested.
Training is done using dataloaders for the train and test sets;
these are custom datasets that select dates between specified start and end dates and return
feature data for that date and a specified number of days and target data for that date alone..
Before training begins the hyperparameters are saved to a json file; 
at specified intevals during training the model is saved to a pkl file and statistics are both printed and are saved to tensorboard.
Model arguments are specified using command line arguments using the argparse functio.

## Model Arguments

For a detailed list of model arguments please read the parse_arguments function in ComNet_Train.py.
Here I will provide a brief overview. The few arguments are file names and locations: 
inputs_file_name, targets_file_name, save_dir, name, version. There is an option for GPU acceleration gpu_accel.

Following this the model hyperparameters are given:
The model configuration is provided as a json string through the argument model_config;
optionally a learning rate scheduler can be implemented by providing a json string for the argument lr_config;
finally there are arguments for the optimizer, learning rate, weight decay, sample noise and batch size.
A table of the model hyperparameters is given below:

| Model Argument | Description | Options |
| --- | --- | --- |
| history_length | length of each channel and the length of feature data fed into the model (given in json string for model_config)| |
| num_inputs | number of input channels (given in json string for model_config)| |
| num_channels | array giving the number of channels between each temporal block; the length of this array gives the depth of the network (given in json string for model_config)| |
| activation | activation function to use in the TCN (given in json string for model_config) | gelu, relu, lrelu |
| kernel_size | kernal size for convolutions (given in json string for model_config) | |
| dropout | proportion of dropout channels, dropout is applied to channels not nodes (given in json string for model_config) | |
| lr_config | json string configuration of the lr scheduler, template lr_config json strings for each method are given in Main Script Experiments notebook| plateau, cosine_warm and one_cycle each with additional specific options |
| optimizer | Optimizer choice | AdamW, SGD, SGD_nesterov |
| lr | Learning rate | |
| weight_decay | Weight decay | |
| sample_noise | Proportional noise to apply to features during training| |
| batch_size | Batch size | |

The user may also specify the number of epochs to train for, the batch interval for printing training stats
and the epoch interval evaluating the model using the test data, saving the model and saving performance stats to tensorboard.

Finally there are some additional arguments that act as switches:

| Switch | Description |
| --- | --- |

| weight_hist | produces histograms for layer parameters at each batch interval which are then saved to tesorboard. |
| print_singular | prints the loss, probability density and MAE on the test data of a gaussian using the average mean and stdev of the model predictions. |
| early_stopping | saves the model at each evaluation and stops if test loss spikes, set as 1 normally or set to restart epoch in restart mode. |
| resume | resume traning of a model from saved version. |





                        
