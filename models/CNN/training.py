import pickle
import numpy as np
import pandas as pd
from keras import models
from keras.layers import Dense, Concatenate
from keras.callbacks import EarlyStopping
import time
import os
import CNN_functions as CNNFuncs

if __name__ == "__main__":

    test_size = 0.2
    random_state = 1

    os.chdir("models/CNN")

    if os.path.exists("models") == False:
        os.mkdir("models")

    # model_version = input("Model Version: ")

    # if os.path.exists("models/{0}".format(model_version)) == False:
    #     os.mkdir("models/{0}".format(model_version))
    # else:
    #     raise ValueError("model version already exists")

    datasets, essential_genes = CNNFuncs.preprocess_data(test_size=test_size, random_state=random_state)

    depmap_datasets = datasets["depmap"]
    RCC_datasets = datasets["RCC"]
    ccRCC_datasets = datasets["ccRCC"]

    top100_essential_genes = essential_genes["top100_essential_genes"]
    top_common_essential_genes = essential_genes["common_essential_genes"]

    expression_data = []
    effect_scores = []

    for gene in top100_essential_genes:

        gene_expression = depmap_datasets[0].copy()
        # gene_expression['Sample_Gene'] = gene_expression.index + "_" + gene
        gene_effect_score = depmap_datasets[2][gene]

        expression_data.append(gene_expression)
        effect_scores.append(gene_effect_score)

    expression_data_df = pd.concat(expression_data).reset_index(drop=True)
    effect_scores_df = pd.concat(effect_scores).reset_index(drop=True)

    print(expression_data_df)

    activation_func = 'relu' 
    activation_func2 = 'linear'
    init = 'he_uniform'
    dense_layer_dim = 250
    batch_size = 500
    num_epoch = 100


    # t = time.time()

    # def createModels(input_dim, intermed_activation_func='relu', final_activation_func='linear', init='he_uniform', trainable=True):

    #     model = models.Sequential()
    #     model.add(Dense(output_dim=500, input_dim=input_dim, activation=intermed_activation_func, init=init))
    #     model.add(Dense(output_dim=250, input_dim=500))
