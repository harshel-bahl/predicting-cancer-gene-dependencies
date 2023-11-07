import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text 
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as mt
from functools import reduce

def CommonCols(df1, df2):
    return df1.columns.intersection(df2.columns)

def CommonIndexes(dataframes):
    return reduce(lambda x, y: x.intersection(y.index), dataframes[1:], dataframes[0].index)

def CombineDFsCol(dfs, column_name, new_column_names):
    
    result = dfs[0][[column_name]].copy()
    
    for df in dfs[1:]:
        result = result.merge(df[[column_name]], left_index=True, right_index=True, how='inner')
    
    result.columns = new_column_names

    return result

def checkGenes(genesCheck, df):

    checkedGenes = {}
    for gene in genesCheck:
        if gene in df.index:
            checkedGenes[gene] = True
        else:
            checkedGenes[gene] = False
    
    return checkedGenes

def z_score_normalize(df, axis=None):
    """Z-score normalize a DataFrame along the specified axis or the entire matrix."""
    
    if axis is None:  # Normalize the entire matrix
        mean = df.values.mean()
        std = df.values.std()
        return (df - mean) / std
    
    mean = df.mean(axis=axis)
    std = df.std(axis=axis)
    
    if axis == 0:  # Normalize along columns
        return (df - mean) / std
    else:  # Normalize along rows
        return df.sub(mean, axis='index').div(std, axis='index')

def min_max_scale_to_minus1_1(df):
    """Apply Min-Max scaling to a DataFrame to transform its values to the range [-1, 1]."""
    X_min = df.min().min()  # Minimum value in the entire DataFrame
    X_max = df.max().max()  # Maximum value in the entire DataFrame
    
    # Check if there's no variance in the dataset
    if X_max == X_min:
        raise ValueError("Cannot apply scaling because max and min values are the same.")
    
    scaled_df = 2 * ((df - X_min) / (X_max - X_min)) - 1
    return scaled_df


def scale_normalize(df, axis=None):
    """Z-score normalize and then scale the DataFrame along the specified axis or the entire matrix."""
    normalized_df = z_score_normalize(df, axis)
    scaled_df = min_max_scale_to_minus1_1(normalized_df)
    return scaled_df

def extract_gene_names(columns):
    return [col.split(' ')[0] for col in columns]

def TopUniqueGenes(dataframes, top_genes_no=100):
    sorted_essential_genes = pd.concat(dataframes).sort_values()

    top_genes_dict = {}
    for gene, score in sorted_essential_genes.items():
        if gene not in top_genes_dict and len(top_genes_dict) < top_genes_no:
            top_genes_dict[gene] = score

    return pd.DataFrame(list(top_genes_dict.items()), columns=['Gene', 'Score'])

def ComputePVals(df1, df2, sig_value=0.05):

    '''Computes P-values for the values in each column (genes) between the dataframes.
    - Requires the columns in both dataframes to be the same'''

    assert all(df1.columns == df2.columns), "Genes in both dataframes must be in the same order"
    
    p_values = []
    for gene in df1.columns:
        _, p_value = ttest_ind(df1[gene], df2[gene], equal_var=False)
        p_values.append(p_value)
    
    _, p_values_adj, _, _ = mt.multipletests(p_values, method='fdr_bh')
    
    result_df = pd.DataFrame({
        'p_val': p_values,
        'q_val(FDR)': p_values_adj
    }, index=df1.columns)
    
    result_df["Significant"] = result_df["q_val(FDR)"] < sig_value

    return result_df

def ComputeDependencyAnalysis(
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        same_score=True, 
        top_thresh=100, 
        return_pVals=False, 
        return_delta_score=False, 
        return_sig_delta_score=False
        ):

    '''
    Inputs:
    - df1, df2: dataframes are expected to be CCLs (rows) and genes (columns), values represent gene effects scores 
    - same_score: set to True if each dataframe is the same gene effects score (like CERES or CHRONOS)
    - top_thresh: Finds the top X no of genes by absolute differences between mean gene effect score of each gene (column)
    - sig_value: probability significance threshold for determining a significant deviation between the gene effect scores for each gene 

    Returns:
    - fil_df1: filtered df1 with the common genes (columns)
    - fil_df2: filtered df2 with the common genes (columns)
    - pVals: pValue dataframe (p-val, q-val FDR, Significant) for each gene computed from the gene effect scores (samples) in each dataframe
    - delta_score: difference in gene effect scores for each gene between the dataframes
    - sig_delta_score: dataframe containing the significant genes sorted based on delta_score
    - sig_delta_score_thresh: dataframe including the top threshold genes from sig_delta_score

    - '''

    commonCols = df1.columns.intersection(df2.columns)

    if same_score:
        fil_df1 = df1[commonCols]
        fil_mean_df1 = fil_df1.mean(axis=0)

        fil_df2 = df2[commonCols]
        fil_mean_df2 = fil_df2.mean(axis=0)

        pVals = ComputePVals(fil_df1, fil_df2)

        sig_cols = pVals[pVals['Significant'] == True]

        delta_score = fil_mean_df1 - fil_mean_df2

        sig_delta_score = delta_score[delta_score.index.isin(sig_cols.index)]
        sig_delta_score.sort_values(ascending=True, inplace=True)

        sig_delta_score_thresh = sig_delta_score[0:top_thresh]

        return_vals = [fil_df1, fil_mean_df1, fil_df2, fil_mean_df2]

        if return_pVals == True:
            return_vals.append(pVals)
        if return_delta_score == True:
            return_vals.append(delta_score)
        if return_sig_delta_score == True:
            return_vals.append(sig_delta_score)
        
        return_vals.append(sig_delta_score_thresh)

        return return_vals
    else:
        return None


def CreateDependencyPlot(
        mean_df1, mean_df2, highlight_df, 
        xAxisTitle,
        yAxisTitle,
        title,
        unhighlightedLabel,
        highlightLabel, 
        figSize=(17.5, 12.5),
        pointSize=5,
        xlim=[-2, 0.5],
        ylim=[-2, 0.5],
        label_no=25,
        ):

    plt.figure(figsize=figSize)

    plt.scatter(mean_df1, mean_df2, s=pointSize, label=unhighlightedLabel, color='gray')
    plt.scatter(mean_df1[highlight_df.index], mean_df2[highlight_df.index], s=pointSize, color='red', label=highlightLabel)

    texts = []
    gene_coords = {}
    for gene in highlight_df.index[:label_no]:
        x = mean_df1.loc[gene]
        y = mean_df2.loc[gene]

        if x > xlim[0] and x < xlim[1] and y > ylim[0] and y < ylim[1]:
            gene_coords[gene] = (x, y)
            texts.append(plt.text(x, y, gene, ha='center', va='center'))

    adjust_text(texts, 
                force_points=(0.5, 0.5), 
                force_text=(0.5, 0.5), 
                expand_points=(1, 1), 
                expand_text=(1, 1))

    for text in texts:
        gene = text.get_text()
        x, y = gene_coords[gene]
        plt.annotate(text.get_text(),
                    xy=(x, y),
                    xytext=(text.get_position()[0], text.get_position()[1]),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    ha='center', va='center')

    plt.title(title)
    plt.xlabel(xAxisTitle)
    plt.ylabel(yAxisTitle)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.grid(True)
    plt.legend()
    plt.show()