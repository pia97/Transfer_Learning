import os
import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

class Data:

    '''For data preprocessing'''

    def __init__(self, data_path, dataset_name, reshape=True):

        self.dataset_name = dataset_name

        # Reading the dataset
        x_train, y_train = self.read_ucr_data(
            data_path+'/'+dataset_name+'/'+dataset_name+'_TRAIN.tsv')
        x_test, y_test = self.read_ucr_data(
            data_path+'/'+dataset_name+'/'+dataset_name+'_TEST.tsv')

        # Number of classes
        self.nb_classes = len(np.unique(y_test))

        # Transform the output classes to start from 0
        y_train = (y_train - y_train.min()) / \
            (y_train.max()-y_train.min())*(self.nb_classes-1)
        y_test = (y_test - y_test.min()) / \
            (y_test.max()-y_test.min())*(self.nb_classes-1)

        # Z-transform
        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        x_train = (x_train - x_train_mean)/(x_train_std)
        x_test = (x_test - x_train_mean)/(x_train_std)

        if reshape == True:
            self.x_train = x_train.reshape(x_train.shape + (1,))
            self.x_test = x_test.reshape(x_test.shape + (1,))
        else:
            self.x_train = x_train
            self.x_test = x_test

        # One-Hot encoding
        self.y_train = keras.utils.to_categorical(y_train, self.nb_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.nb_classes)

    def read_ucr_data(self, filename):
        # add delimiter=',' for Archive 2015 
        data = np.loadtxt(filename)
        Y = data[:, 0]
        X = data[:, 1:]
        return X, Y

    def get_x_train(self):
        return self.x_train

    def get_y_train(self):
        return self.y_train

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test

    def get_nb_classes(self):
        return self.nb_classes

    def get_data_set_name(self):
        return self.dataset_name   

    # get input shape

def save_results_csv(file_name,dataset,model,training_time,y_true,y_pred):
    
    res = pd.DataFrame(data=np.zeros((1, 7), dtype=np.float), index=[0],
                       columns=['dataset','model', 'training_time','accuracy','precision', 'recall','f1-score'])
    
    res['dataset'] = dataset
    res['model'] = model
    res['training_time'] = training_time

    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')   
    res['f1-score'] = f1_score(y_true, y_pred, average='macro')  
 
    res.to_csv(file_name+'.csv', index=False)

    return res    


