import pickle
import numpy as np
import pandas as pd
from keras import models
from keras.layers import Dense, Concatenate
from keras.callbacks import EarlyStopping

if __name__ == "__main__":
    
    with open('DeepDEP/code/data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    fprint = pd.read_csv('DeepDEP/code/data/crispr_gene_fingerprint_cgp.txt', sep="\t")
    fprint.set_index('GeneSet', inplace=True)
    print(fprint)

    print(data_mut.shape)
    # print(pd.DataFrame(data_exp))
    # print(data_dep)

    # geneExp = pd.read_csv('DeepDEP/code/data/ccl_exp_data_paired_with_tcga.txt', sep="\t")
    # print(geneExp)