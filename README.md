# ComNet

ComNet is a TCN designed to predict the returns on commodity pricing data based on any number of timeseries input channels,
typically prices or indicators of the commodity itself along with related tickers such as other commodities, Forex, Bond yields etc.

## Inputs

I have provided some raw inputs in the Data folder. These inputs need to be processed before they can be used by the network, this is done in Financial_Data_Processing.
Processing the data produces different feature dataframes, for example for different subsets of tickers or using technical indicators instead of price data.
Different target dataframes are also produced, for example those for different commodities or those to match the technical indicator feature dataframes, in which some initial dates are omitted.
Before the data is saved any price data is converted into a percentage return including the target price. All the Input data is saved under Data/Inputs.
