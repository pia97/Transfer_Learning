# Encoder, adapted from Fawaz et. al's implementation
# https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py
#
# Network originally proposed by:
#
# @article{serra2018towards,
#   title={Towards a universal neural network encoder for time series},
#   author={Serr{\`a}, J and Pascual, S and Karatzoglou, A},
#   journal={Artif Intell Res Dev Curr Chall New Trends Appl},
#   volume={308},
#   pages={120},
#   year={2018}
# }

from classifiers.cnn import Cnn 
from tensorflow import keras
import numpy as np
import pandas as pd
from utils.utils import Data
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score

class Encoder(Cnn):
    def build_model(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input_layer = keras.layers.Input(self.x_train.shape[1:])
        
            # conv block -1
            conv1 = keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(input_layer)
            conv1 = tfa.layers.InstanceNormalization()(conv1)
            conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
            conv1 = keras.layers.Dropout(rate=0.2)(conv1)
            conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        
             # conv block -2
            conv2 = keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
            conv2 = tfa.layers.InstanceNormalization()(conv2)
            conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
            conv2 = keras.layers.Dropout(rate=0.2)(conv2)
            conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        
            # conv block -3
            conv3 = keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
            conv3 = tfa.layers.InstanceNormalization()(conv3)
            conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
            conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        
             # split for attention
            attention_data = keras.layers.Lambda(lambda x: x[:,:,:256])(conv3)
            attention_softmax = keras.layers.Lambda(lambda x: x[:,:,256:])(conv3)
        
             # attention mechanism
            attention_softmax = keras.layers.Softmax()(attention_softmax)
            multiply_layer = keras.layers.Multiply()([attention_softmax,attention_data])
      
            # last layer
            dense_layer = keras.layers.Dense(units=256,activation='sigmoid')(multiply_layer)
            dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
        
            # Fully connected layer
            flatten_layer = keras.layers.Flatten()(dense_layer)
            output_layer = keras.layers.Dense(units=self.nb_classes,activation='softmax')(flatten_layer)

            self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
            optimizer = keras.optimizers.Adam(0.00001)
            self.model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

            # self.model.summary()  

    def retrain( self, pre_trained_directory, epochs, batch_size, weight_name, file_path, num_layers=0, ):

        input_file = pre_trained_directory + '/' + weight_name +'/' + weight_name + '_Encoder_best_model.hdf5'

        old = keras.models.load_model( input_file )
        weights = [ layer.get_weights() for layer in old.layers ]

        for i in range( 0, ( len( self.model.layers ) - 1 ) ):
            self.model.layers[i].set_weights( weights[ i ] )

        for i in range( 0, num_layers ):
            self.model.layers[i].trainable = False

        self.model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau( monitor='loss', factor=0.5, patience=50, min_lr=0.0001 )
        outfile = self.output_directory + '/Encoder_' + self.data_set_name + "_Transfer_" + weight_name + "_" + str( epochs ) + '_' + str(num_layers)
        
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
            fd.write( 'ENCODER;' + weight_name + ';' + self.data_set_name + ';' + str( num_layers ) + ';' + str( epochs ) + ';' + str( self.accuracy ) + '\n' )
            fd.close()

        keras.backend.clear_session()