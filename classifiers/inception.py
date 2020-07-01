# InceptionTime, adapted from Fawaz et. al's implementation
# https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
#
# @article{IsmailFawaz2019inceptionTime,
#   Title                    = {InceptionTime: Finding AlexNet for Time Series Classification},
#   Author                   = {Ismail Fawaz, Hassan and Lucas, Benjamin and Forestier, Germain and Pelletier, Charlotte and Schmidt, Daniel F. and Weber, Jonathan and Webb, Geoffrey I. and Idoumghar, Lhassane and Muller, Pierre-Alain and Petitjean, Francois},
#   journal                  = {ArXiv},
#   Year                     = {2019}
# }

from classifiers.cnn import Cnn 
from tensorflow import keras
import numpy as np
import pandas as pd
from utils.utils import Data
import tensorflow as tf 
from sklearn.metrics import accuracy_score

class Inception(Cnn):
    def __init__(self, data, output_directory, 
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41):
        super(Inception, self).__init__(data, output_directory)
        
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = 32
        
        
    

    def _inception_module(self, input_tensor, stride=1, activation='linear'):
    
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self):

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input_layer = keras.layers.Input(self.x_train.shape[1:])
        

            x = input_layer
            input_res = input_layer

            for d in range(self.depth):

                x = self._inception_module(x)

                if self.use_residual and d % 3 == 2:
                    x = self._shortcut_layer(input_res, x)
                    input_res = x

            gap_layer = keras.layers.GlobalAveragePooling1D()(x)

            output_layer = keras.layers.Dense(self.nb_classes, activation='softmax')(gap_layer)

            self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

            self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])


        #self.model.summary()

    
    def retrain( self, pre_trained_directory, epochs, batch_size, weight_name, file_path, num_layers=0, ):

        input_file = pre_trained_directory + '/' + weight_name +'/' + weight_name + '_Inception_best_model.hdf5'

        old = keras.models.load_model( input_file )
        weights = [ layer.get_weights() for layer in old.layers ]

        for i in range( 0, ( len( self.model.layers ) - 1 ) ):
            self.model.layers[i].set_weights( weights[ i ] )

        for i in range( 0, num_layers ):
            self.model.layers[i].trainable = False

        self.model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau( monitor='loss', factor=0.5, patience=50, min_lr=0.0001 )
        outfile = self.output_directory + '/Inception_' + self.data_set_name + "_Transfer_" + weight_name + "_" + str( epochs ) + '_' + str(num_layers)
        
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
            fd.write( 'INCEPTION;' + weight_name + ';' + self.data_set_name + ';' + str( num_layers ) + ';' + str( epochs ) + ';' + str( self.accuracy ) + '\n' )
            fd.close()

        keras.backend.clear_session()