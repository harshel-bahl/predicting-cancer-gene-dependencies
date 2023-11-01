import numpy as np
import pandas as pd

def preprocess_depmap_data():

    ### Selecting Kidney Cancer Cell Lines from DepMap
    ##################################################

    depmap_rcc = pd.read_csv('/Users/harshel/Documents/harvard/BMI751_capstone/datasets/DepMap_datasets/23Q2/kidney_cancer_list.txt')
    depmap_rcc.columns = ["Cell Line", "Lineage Subtype", "DepMapID"]
    depmap_rcc = depmap_rcc.map(lambda x: x.strip() if isinstance(x, str) else x)
    depmap_rcc.set_index("Cell Line", inplace=True)

    depmap_rcc_IDs = depmap_rcc["DepMapID"].to_list()

    print("Number of DepMap Kidney Cancer CCLs: {0}\n".format(len(depmap_rcc_IDs)))
    # print(depmap_rcc)

    ### Filtering for selected Kidney Cancer Cell Lines from DepMap
    ###############################################################

    depmap_gene_exp_23Q2 = pd.read_csv('/Users/harshel/Documents/harvard/BMI751_capstone/datasets/DepMap_datasets/23Q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv')
    depmap_gene_exp_23Q2.set_index("Unnamed: 0", inplace=True)
    depmap_gene_exp_23Q2.index.name = "Cell Line"

    aggregated_depmap_gene_exp = depmap_gene_exp_23Q2[depmap_gene_exp_23Q2.index.isin(depmap_rcc_IDs)]
    aggregated_depmap_gene_exp.sort_index(axis=1, inplace=True)

    # print(aggregated_depmap_gene_exp)

    missing_depmapIDs = [depmapID for depmapID in depmap_rcc_IDs if depmapID not in aggregated_depmap_gene_exp.index.to_list()]
    missing_CCLs = []
    for depmapID in missing_depmapIDs: 
        missing_CCLs.append(depmap_rcc[depmap_rcc["DepMapID"] == depmapID].index[0])

    print("\nMissing RCC DepMapIDs from aggregated database: \n{0}\n".format(missing_CCLs))

    ### CRISPR Gene Effect Scores Preprocessing
    ###########################################

    depmap_gene_effect_23Q2 = pd.read_csv('/Users/harshel/Documents/harvard/BMI751_capstone/datasets/DepMap_datasets/23Q2/CRISPRGeneEffect.csv')
    depmap_gene_effect_23Q2.set_index("ModelID", inplace=True)
    depmap_gene_effect_23Q2.index.name = "Cell Line"
    depmap_gene_effect_23Q2.sort_index(axis=1, inplace=True)

    aggregated_depmap_gene_effects = depmap_gene_effect_23Q2[depmap_gene_effect_23Q2.index.isin(depmap_rcc_IDs)]

    # print(aggregated_depmap_gene_effects)

    ### Preprocess Common DepMap Indices
    ####################################

    common_depmap_indices = aggregated_depmap_gene_exp.index.intersection(aggregated_depmap_gene_effects.index)

    processed_depmap_gene_exp = aggregated_depmap_gene_exp[aggregated_depmap_gene_exp.index.isin(common_depmap_indices)].sort_index(axis=0)
    processed_depmap_gene_effects = aggregated_depmap_gene_effects[aggregated_depmap_gene_effects.index.isin(common_depmap_indices)].sort_index(axis=0)

    return processed_depmap_gene_exp, processed_depmap_gene_effects

preprocess_depmap_data()


def preprocess_tRCC_data():

    ### Load tRCC Gene Effects
    ##########################

    FUUR1 = pd.read_csv("../datasets/tRCC_cell_lines/MAGeCK_output/FUUR1_d28_vs_pDNA.gene_summary.txt", sep="\t").set_index('id', inplace=False)
    STFE = pd.read_csv("../datasets/tRCC_cell_lines/MAGeCK_output/STFE_d28_vs_pDNA.gene_summary.txt", sep="\t").set_index('id', inplace=False)
    UOK109 = pd.read_csv("../datasets/tRCC_cell_lines/MAGeCK_output/UOK109_d28_vs_pDNA.gene_summary.txt", sep="\t").set_index('id', inplace=False)
    UOK146 = pd.read_csv("../datasets/tRCC_cell_lines/MAGeCK_output/UOK146_d28_vs_pDNA.gene_summary.txt", sep="\t").set_index('id', inplace=False)


preprocess_tRCC_data()


