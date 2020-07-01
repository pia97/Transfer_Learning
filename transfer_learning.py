from classifiers.cnn import Cnn
from classifiers.resnet import Resnet
from classifiers.encoder import Encoder
from classifiers.inception import Inception
from utils.utils import Data
import pandas as pd
import os
import sys
from tensorflow import keras

# Input data directory, for UCR archive 128  
input_directory = '/home/pia/Dokumente/Studium/AI/UCRArchive_2018'

# Results directory 
output_directory = '/home/pia/Dokumente/Studium/AI//Transfer Learning Server/results_transfer'

# Input data directory, for pretrained models
pre_trained_directory = '/home/pia/Dokumente/Studium/AI/tl-results'

# List of the datasets 
dataset_names = pd.read_csv('DataSummary.csv')  
UCR128 = dataset_names['Name']

res = pd.DataFrame()
dataset_names = ['Adiac','Car','Coffee']
models = ['CNN', 'RESNET', 'ENCODER', 'INCEPTION']
epochs = [20, 100, 500, 1000]
frozen_layers_Cnn = [0, 4, 7, 11]
frozen_layers_Resnet = [0, 13, 25, 37]
frozen_layers_Encoder = [0, 5, 11, 19, 22]
frozen_layers_Inception = [0, 10, 19, 32, 41, 50, 63]

#epochs = [1, 2]
#frozen_layers_Cnn = [10, 11]
#frozen_layers_Resnet = [36, 37]
#frozen_layers_Encoder = [21, 22]
#frozen_layers_Inception = [62, 63]

gpu = 4

file_path = output_directory + '/results.csv'

with open( file_path, 'w' ) as fd:
    fd.write( 'model;pre_trained_set;training_set;num_frozen_layers;epochs;accuracy\n' )
    fd.close()

for dataset_name in UCR128:
#for dataset_name in dataset_names:

    # Ouput directory for each dataset
    output_dataset = output_directory+'/'+dataset_name+'/'

    for m in models:
        path = output_dataset + '/' + m

        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory ", path,  " Created ")
        else:
            print("Directory ", path,  " already exists")

    data_instance = Data(input_directory, dataset_name, reshape=True)

    for transfer_set in UCR128:
    #for transfer_set in dataset_names:

        if transfer_set != dataset_name:

            for eps in epochs:

                for num_layers in frozen_layers_Cnn:

                    cnn = Cnn( data_instance, output_dataset + '/CNN' )
                    cnn.build_model()
                    cnn.retrain( pre_trained_directory, eps, 16 * gpu, transfer_set, file_path, num_layers )


                for num_layers in frozen_layers_Resnet:

                    res = Resnet( data_instance, output_dataset + '/RESNET' )
                    res.build_model()
                    res.retrain( pre_trained_directory, eps, 16 * gpu, transfer_set, file_path, num_layers=num_layers )


                for num_layers in frozen_layers_Encoder:

                    enc = Encoder( data_instance, output_dataset + '/ENCODER' )
                    enc.build_model()
                    enc.retrain( pre_trained_directory, eps, 12, transfer_set, file_path, num_layers=num_layers )


                for num_layers in frozen_layers_Inception:

                    inc = Inception( data_instance, output_dataset + '/INCEPTION' )
                    inc.build_model()
                    inc.retrain( pre_trained_directory, eps, 64 * gpu, transfer_set, file_path, num_layers=num_layers )
