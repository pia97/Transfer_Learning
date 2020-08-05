from classifiers.cnn import Cnn
from classifiers.resnet import Resnet
from classifiers.encoder import Encoder
from classifiers.inception import Inception
from utils.utils import Data
import pandas as pd
import os
import sys
from tensorflow import keras
import tensorflow as tf

# initialization should not allocate all memory on the device
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            break
    except RuntimeError as e:
        print(e)

# Input data directory, for UCR archive 128
#input_directory = '/home/alamayreh/UCRArchive_2018'
input_directory = '/home/pia/Dokumente/Studium/AI/UCRArchive_2018'


# Results directory
#output_directory = '/home/alamayreh/pia_resutls/results_transfer'
output_directory = '/home/pia/Dokumente/Studium/AI/Transfer_Learning/results'

# Input data directory, for pretrained models
#pre_trained_directory = '/home/alamayreh/results_training_backup'
pre_trained_directory = '/home/pia/Dokumente/Studium/AI/tl-results'

# List of the datasets
#dataset_names = pd.read_csv('DataSummary.csv')
#UCR128 = dataset_names['Name']
dataset_names = pd.read_csv('similar.csv')
UCR86 = dataset_names['Name']

transfered = dataset_names.T
transfered.columns = transfered.iloc[0]
transfered = transfered.drop( 'Name' )

res = pd.DataFrame()
#dataset_names = ['Adiac']
#transfer_names = ['DiatomSizeReduction', 'SwedishLeaf', 'ShapesAll']
models = ['CNN', 'RESNET', 'ENCODER', 'INCEPTION']
#epochs = [20, 100, 500, 1000]
#frozen_layers_Cnn = [0, 4, 7, 11]
#frozen_layers_Resnet = [0, 13, 25, 37]
#frozen_layers_Encoder = [0, 5, 11, 19, 22]
#frozen_layers_Inception = [0, 10, 19, 32, 41, 50, 63]


epochs = [1]
frozen_layers_Cnn = [11]
frozen_layers_Resnet = [37]
frozen_layers_Encoder = [22]
frozen_layers_Inception = [63]

gpu = 1

file_path = output_directory + '/results.csv'

with open(file_path, 'w') as fd:
    fd.write('model;pre_trained_set;training_set;num_frozen_layers;epochs;accuracy\n')
    fd.close()

"""
with tf.device('/device:GPU:1'):
    for dataset_name in UCR86:
        # for dataset_name in dataset_names:

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

        for transfer_set in transfer_names:
            # for transfer_set in dataset_names:

            if transfer_set != dataset_name:

                for eps in epochs:

                    for num_layers in frozen_layers_Cnn:

                        cnn = Cnn(data_instance, output_dataset + '/CNN')
                        cnn.build_model()
                        cnn.retrain(pre_trained_directory, eps, 16 *
                                    gpu, transfer_set, file_path, num_layers)

                    for num_layers in frozen_layers_Resnet:

                        res = Resnet(data_instance, output_dataset + '/RESNET')
                        res.build_model()
                        res.retrain(pre_trained_directory, eps, 16 * gpu,
                                    transfer_set, file_path, num_layers=num_layers)

                    for num_layers in frozen_layers_Encoder:

                        enc = Encoder(data_instance, output_dataset + '/ENCODER')
                        enc.build_model()
                        enc.retrain(pre_trained_directory, eps, 12,
                                    transfer_set, file_path, num_layers=num_layers)

                    for num_layers in frozen_layers_Inception:

                        inc = Inception(
                            data_instance, output_dataset + '/INCEPTION')
                        inc.build_model()
                        inc.retrain(pre_trained_directory, eps, 64 * gpu,
                                    transfer_set, file_path, num_layers=num_layers)
"""
for dataset_name in UCR86:
    # for dataset_name in dataset_names:

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

    transfer_names = transfered[ dataset_name ]
    for transfer_set in transfer_names:
        # for transfer_set in dataset_names:

        if transfer_set != dataset_name:

            for eps in epochs:

                for num_layers in frozen_layers_Cnn:

                    cnn = Cnn(data_instance, output_dataset + '/CNN')
                    cnn.build_model()
                    cnn.retrain(pre_trained_directory, eps, 16 *
                                gpu, transfer_set, file_path, num_layers)

                for num_layers in frozen_layers_Resnet:

                    res = Resnet(data_instance, output_dataset + '/RESNET')
                    res.build_model()
                    res.retrain(pre_trained_directory, eps, 16 * gpu,
                                transfer_set, file_path, num_layers=num_layers)

                for num_layers in frozen_layers_Encoder:

                    enc = Encoder(data_instance, output_dataset + '/ENCODER')
                    enc.build_model()
                    enc.retrain(pre_trained_directory, eps, 12,
                                    transfer_set, file_path, num_layers=num_layers)

                for num_layers in frozen_layers_Inception:

                    inc = Inception(
                        data_instance, output_dataset + '/INCEPTION')
                    inc.build_model()
                    inc.retrain(pre_trained_directory, eps, 64 * gpu,
                                transfer_set, file_path, num_layers=num_layers)