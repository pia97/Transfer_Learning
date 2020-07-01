# Residual network, adapted from Fawaz et. al's implementation
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

from classifiers.cnn import Cnn 
from tensorflow import keras
import numpy as np
import pandas as pd
from utils.utils import Data
import tensorflow as tf
from sklearn.metrics import accuracy_score

class Resnet(Cnn):


    def build_model(self):
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            n_feature_maps = 64

            input_layer = keras.layers.Input(self.x_train.shape[1:])  

            # BLOCK 1

            conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)

            conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)

            conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)

            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            output_block_1 = keras.layers.add([shortcut_y, conv_z])
            output_block_1 = keras.layers.Activation('relu')(output_block_1)

            # BLOCK 2

            conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)

            conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)

            conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)

            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)

            # BLOCK 3

            conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)

            conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)

            conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)

            # no need to expand channels because they are equal
            shortcut_y = keras.layers.BatchNormalization()(output_block_2)

            output_block_3 = keras.layers.add([shortcut_y, conv_z])
            output_block_3 = keras.layers.Activation('relu')(output_block_3)

            # Global Average Pooling layer

            gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

            # Fully connected layer
            output_layer = keras.layers.Dense(self.nb_classes, activation='softmax')(gap_layer)

            self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
            optimizer = keras.optimizers.Adam()
            self.model.compile(loss='categorical_crossentropy',
                optimizer=optimizer, metrics=['accuracy'])


    def retrain( self, pre_trained_directory, epochs, batch_size, weight_name, file_path, num_layers=0, ):

        input_file = pre_trained_directory + '/' + weight_name +'/' + weight_name + '_Resnet_best_model.hdf5'

        old = keras.models.load_model( input_file )
        weights = [ layer.get_weights() for layer in old.layers ]

        for i in range( 0, ( len( self.model.layers ) - 1 ) ):
            self.model.layers[i].set_weights( weights[ i ] )

        for i in range( 0, num_layers ):
            self.model.layers[i].trainable = False

        self.model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau( monitor='loss', factor=0.5, patience=50, min_lr=0.0001 )
        outfile = self.output_directory + '/Resnet_' + self.data_set_name + "_Transfer_" + weight_name + "_" + str( epochs ) + '_' + str(num_layers)
        
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
            fd.write( 'RESNET;' + weight_name + ';' + self.data_set_name + ';' + str( num_layers ) + ';' + str( epochs ) + ';' + str( self.accuracy ) + '\n' )
            fd.close()

        keras.backend.clear_session()