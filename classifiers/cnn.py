# Fully 3-layers convolutional neural network, adapted from from Fawaz et. al's implementation
# https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
#
# Network originally proposed by:
#
# @inproceedings{wang2017time,
#   title={Time series classification from scratch with deep neural networks: A strong baseline},
#   author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
#   booktitle={2017 International joint conference on neural networks (IJCNN)},
#   pages={1578--1585},
#   year={2017},
#   organization={IEEE}
# }


from time import time
from tensorflow import keras

import numpy as np
import pandas as pd

from utils.utils import Data
from utils.utils import save_results_csv

import tensorflow as tf
from sklearn.metrics import accuracy_score

__metaclass__ = type


class Cnn:

    def __init__( self, data, output_directory ):

        self.output_directory = output_directory
        self.training_time = 0
        self.batch_size = 16

        self.x_train = data.get_x_train()
        self.y_train = data.get_y_train()
        self.x_test = data.get_x_test()
        self.y_test = data.get_y_test()

        self.nb_classes = data.get_nb_classes()
        self.data_set_name = data.get_data_set_name()


    def build_model( self ):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input_layer = keras.layers.Input(self.x_train.shape[1:])
            # Conv layer 1
            conv1 = keras.layers.Conv1D(
                filters=128, kernel_size=8, padding='same')(input_layer)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)

            # Conv layer 2
            conv2 = keras.layers.Conv1D(
                filters=256, kernel_size=5, padding='same')(conv1)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)

            # Conv layer 3
            conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3)

            # Global Average Pooling layer
            full = keras.layers.GlobalAveragePooling1D()(conv3)

            # Fully connected layer
            out = keras.layers.Dense(self.nb_classes, activation='softmax')(full)

            self.model = keras.models.Model(inputs=input_layer, outputs=out)

            optimizer = keras.optimizers.Adam()
            self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer, metrics=['accuracy'])

            #self.model.summary()


    def retrain( self, pre_trained_directory, epochs, batch_size, weight_name, file_path, num_layers=0, ):

        input_file = pre_trained_directory + '/' + weight_name +'/' + weight_name + '_Cnn_best_model.hdf5'

        old = keras.models.load_model( input_file )
        weights = [ layer.get_weights() for layer in old.layers ]

        for i in range( 0, ( len( self.model.layers ) - 1 ) ):
            self.model.layers[i].set_weights( weights[ i ] )

        for i in range( 0, num_layers ):
            self.model.layers[i].trainable = False

        self.model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


        reduce_lr = keras.callbacks.ReduceLROnPlateau( monitor='loss', factor=0.5, patience=50, min_lr=0.0001 )
        outfile = self.output_directory + '/Cnn_' + self.data_set_name + "_Transfer_" + weight_name + "_" + str( epochs ) + '_' + str(num_layers)
        
        model_checkpoint = keras.callbacks.ModelCheckpoint( filepath = outfile + "_best_model.hdf5", monitor = 'loss', save_best_only=True )

        hist = self.model.fit( self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data = (self.x_test, self.y_test), callbacks=[reduce_lr, model_checkpoint] )

        np.save( outfile + '_history.npy', hist.history )
        self.model.save( outfile + '_last_model.hdf5' )

        self.model = keras.models.load_model( outfile + "_best_model.hdf5" )
        predicted = self.model.predict( self.x_test )
        predicted = np.argmax( predicted, axis = 1 )
        predicted = keras.utils.to_categorical(predicted, self.nb_classes)

        self.accuracy = accuracy_score( self.y_test, predicted )

        with open( file_path, 'a' ) as fd:
            fd.write( 'CNN;' + weight_name + ';' + self.data_set_name + ';' + str( num_layers ) + ';' + str( epochs ) + ';' + str( self.accuracy ) + '\n' )
            fd.close()

        keras.backend.clear_session()

