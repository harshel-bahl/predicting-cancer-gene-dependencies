import pickle
import numpy as np
import pandas as pd
from keras import models
from keras.layers import Dense, Concatenate
from keras.callbacks import EarlyStopping
import time
import process_data as pData

if __name__ == "__main__":

    '''
    - Preprocess data
    - 
    '''

    dataset = pData.preprocess_depmap_data()

    '''Gene Expression Model
    - ~35 RCC cancer cell lines
    - ~19000 genes as predictors
    '''

    ### Model1: Gene Exp
    t = time.time()

    def createModels(input_dim, intermed_activation_func='relu', final_activation_func='linear', init='he_uniform', trainable=True):

        model = models.Sequential()
        model.add(Dense(output_dim=500, input_dim=input_dim, activation=intermed_activation_func, init=init))
        model.add(Dense(output_dim=250, input_dim=500))



    # geneExp = pd.read_csv('DeepDEP/code/data/ccl_exp_data_paired_with_tcga.txt', sep="\t")
    # print(geneExp)