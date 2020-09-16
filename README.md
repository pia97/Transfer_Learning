# Transfer_Learning
This repository contains the code used for executing an experiment in the field of transfer learning for time series classification.
It traines four different network architecutres (CNN, ResNet, Encoder, Inception) on the 85 datasets provided by the UCR Archive with using the most similar dataset as pre-trained network as given by the paper "Transfer learning for time series classification".

### Python verion
This codes runs on Python 2. Also the library tensorflow must be installed before using this code

### Dataset
It's necessary to download the UCR Archive (https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) which contains 128 datasets from which 85 are used. The path in the code as well as the path were the results should be saved are to be changed before running the code.
The data on the similar datasets are stored in the file similar.csv.

### Variables
The number of epochs and the number of frozen layers for each architecture to use in the experiment can be changed through the coresponding variables. The code will train one network for each combination of these for each dataset.

### Running the code
For running the code the script run.sh is executed. The resulting model accuracies are stored in the file results.csv in the given directory
