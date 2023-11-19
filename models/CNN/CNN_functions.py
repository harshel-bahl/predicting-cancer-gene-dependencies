import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def ExtractGenesNames(columns):
    return [col.split(' ')[0] for col in columns]

def Intersect_DF(dfs, match="both"):

    if not dfs:
        raise ValueError("The list of dataframes is empty")

    common_index = dfs[0].index
    common_columns = dfs[0].columns

    for df in dfs[1:]:
        if match in ("rows", "both"):
            common_index = np.intersect1d(common_index, df.index)
        if match in ("columns", "both"):
            common_columns = np.intersect1d(common_columns, df.columns)

    intersected_dfs = []
    for df in dfs:
        if match == "both":
            df_intersected = df.loc[common_index, common_columns]
        elif match == "rows":
            df_intersected = df.loc[common_index]
        elif match == "columns":
            df_intersected = df[common_columns]
        intersected_dfs.append(df_intersected)

    return intersected_dfs

def Check_DF_Similarity(dfs, check='both'):

    if not dfs or not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise ValueError("Please provide a list of pandas DataFrames.")

    if check not in ['columns', 'rows', 'both']:
        raise ValueError("Check argument must be 'columns', 'rows', or 'both'")

    def get_differences(sets):
        common_elements = set.intersection(*sets)
        return [list(common_elements.symmetric_difference(s)) for s in sets]

    columns_sets = [set(df.columns) for df in dfs]
    rows_sets = [set(df.index) for df in dfs]

    if check == 'columns':
        differences = get_differences(columns_sets)
        all_match = all(len(diff) == 0 for diff in differences)
        return {'all_match': all_match, 'column_differences': differences}

    elif check == 'rows':
        differences = get_differences(rows_sets)
        all_match = all(len(diff) == 0 for diff in differences)
        return {'all_match': all_match, 'row_differences': differences}

    elif check == 'both':
        col_diff = get_differences(columns_sets)
        row_diff = get_differences(rows_sets)
        all_match = all(len(diff) == 0 for diff in col_diff) and all(
            len(diff) == 0 for diff in row_diff)
        return {'all_match': all_match, 'column_differences': col_diff, 'row_differences': row_diff}
    
def preprocess_data(test_size, random_state, mode="depmap", extract_top_genes=5000, extract_neg_selec=True, check_missing_essential_genes=True):

    ### Essential Genes
    ###################
    top_common_essential_genes = pd.read_csv("../../analysis/testGenes/top_common_essential_genes")["gene"]
    top100_essential_genes = pd.read_csv("../../analysis/testGenes/top_essential_genes")["Gene"]

    ### DepMap Gene Expression (TPMLogp1)
    #####################################
    depmap_gene_exp_23Q2 = pd.read_csv("../../datasets/depmap_datasets/23Q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    depmap_gene_exp_23Q2.set_index("Unnamed: 0", inplace=True)
    depmap_gene_exp_23Q2.index.name = None
    depmap_gene_exp_23Q2.sort_index(axis=0, inplace=True)
    depmap_gene_exp_23Q2.sort_index(axis=1, inplace=True)
    depmap_gene_exp_23Q2.dropna(axis=1, inplace=True)
    depmap_gene_exp_23Q2.columns = ExtractGenesNames(depmap_gene_exp_23Q2.columns)

    ### DepMap Gene Effects (CHRONOS)
    #################################
    depmap_gene_effect_23Q2 = pd.read_csv('../../datasets/depmap_datasets/23Q2/CRISPRGeneEffect.csv')
    depmap_gene_effect_23Q2.set_index("ModelID", inplace=True)
    depmap_gene_effect_23Q2.index.name = None
    depmap_gene_effect_23Q2.sort_index(axis=0, inplace=True)
    depmap_gene_effect_23Q2.sort_index(axis=1, inplace=True)
    depmap_gene_effect_23Q2.dropna(axis=1, inplace=True)
    depmap_gene_effect_23Q2.columns = ExtractGenesNames(depmap_gene_effect_23Q2.columns)

    ### TRCC Gene Expression (TPMLogp1)
    ###################################
    DFCI_gene_exp = pd.read_csv("../../datasets/tRCC_cell_lines/raw/RSEM_summary_all_samples_gene_TPM.txt", sep="\t").set_index("gene_id")

    STFE1 = pd.read_csv("../../datasets/tRCC_cell_lines/raw/F1.genes.results", sep="\t").set_index("gene_id")["TPM"]
    STFE2 = pd.read_csv("../../datasets/tRCC_cell_lines/raw/F2.genes.results", sep="\t").set_index("gene_id")["TPM"]
    STFE3 = pd.read_csv("../../datasets/tRCC_cell_lines/raw/F3.genes.results", sep="\t").set_index("gene_id")["TPM"]
    STFE_means = pd.concat([STFE1, STFE2, STFE3], axis=1).dropna(axis=1).mean(axis=1)
    STFE_means.name = "STFE"

    FUUR1_means = DFCI_gene_exp[['B19', 'B20', 'B21']].mean(axis=1)
    UOK109_means = DFCI_gene_exp[['B10', 'B11', 'B12']].mean(axis=1)

    tRCC_gene_exp = pd.DataFrame({
        'FUUR1': FUUR1_means,
        'UOK109': UOK109_means
    }, index=DFCI_gene_exp.index)

    tRCC_gene_exp = tRCC_gene_exp.join(STFE_means, how="outer")

    tRCC_gene_exp.index.name = None
    tRCC_gene_exp.index = tRCC_gene_exp.index.str.split('_').str[-1]
    tRCC_gene_exp.sort_index(axis=0, inplace=True)
    tRCC_gene_exp.sort_index(axis=1, inplace=True)
    tRCC_gene_exp = np.log1p(tRCC_gene_exp)
    tRCC_gene_exp = tRCC_gene_exp.groupby(tRCC_gene_exp.index).sum().T
    tRCC_gene_exp.to_csv("../../datasets/tRCC_cell_lines/tRCC_gene_exp_TPMLogp1.csv")

    ### TRCC Gene Effects (CHRONOS)
    ###############################
    DFCI_chronos_dataset = pd.read_csv("../../datasets/tRCC_cell_lines/Chronos/tRCC_chronos_summary_for_BL_ASPS_updated.csv")
    DFCI_chronos_CCLs = DFCI_chronos_dataset[["Gene", "PC3", "CAKI2", "CAKI1", "786O", "DU145", "HCT116", "NCIH460", "FUUR1", "STFE", "UOK109"]].T
    DFCI_chronos_CCLs.columns = DFCI_chronos_CCLs.iloc[0]
    DFCI_chronos_CCLs.columns.name = None
    DFCI_chronos_CCLs.drop(DFCI_chronos_CCLs.index[0], inplace=True)
    DFCI_chronos_CCLs = DFCI_chronos_CCLs.loc[:, ~(DFCI_chronos_CCLs == 'Unknown').any(axis=0)].apply(pd.to_numeric, errors='coerce').dropna(axis=1)

    tRCC_chronos_gene_effects = DFCI_chronos_CCLs.loc[["FUUR1", "STFE", "UOK109"]]
    tRCC_chronos_gene_effects.sort_index(axis=0, inplace=True)
    tRCC_chronos_gene_effects.sort_index(axis=1, inplace=True)

    ### Standardise Rows and Columns Between dataframes
    ###################################################
    depmap_gene_exp_23Q2, depmap_gene_effect_23Q2 = Intersect_DF([depmap_gene_exp_23Q2, depmap_gene_effect_23Q2])
    tRCC_gene_exp, tRCC_chronos_gene_effects = Intersect_DF([tRCC_gene_exp, tRCC_chronos_gene_effects])
    depmap_gene_exp_23Q2, depmap_gene_effect_23Q2, tRCC_gene_exp, tRCC_chronos_gene_effects = Intersect_DF([depmap_gene_exp_23Q2, depmap_gene_effect_23Q2, tRCC_gene_exp, tRCC_chronos_gene_effects], match="columns")

    depmap_check = Check_DF_Similarity([depmap_gene_exp_23Q2, depmap_gene_effect_23Q2])
    tRCC_check = Check_DF_Similarity([tRCC_gene_exp, tRCC_chronos_gene_effects])
    depmap_tRCC_check = Check_DF_Similarity([depmap_gene_exp_23Q2, depmap_gene_effect_23Q2, tRCC_gene_exp, tRCC_chronos_gene_effects], check="columns")

    print("DepMap dataset row/column check: ", depmap_check)
    print("tRCC dataset row/column check: ", tRCC_check)
    print("Check columns across DepMap and tRCC dataframes: ", depmap_tRCC_check)

    if depmap_check["all_match"] != True or tRCC_check["all_match"] != True or depmap_tRCC_check["all_match"] != True:
        raise ValueError("datasets don't match rows and/or columns")

    if extract_top_genes != None:
        if extract_neg_selec:
            top_extracted_genes = depmap_gene_effect_23Q2.mean(axis=0).sort_values(ascending=True).head(extract_top_genes).index
        else:
            top_extracted_genes = abs(depmap_gene_effect_23Q2).mean(axis=0).sort_values(ascending=False).head(extract_top_genes).index

        if check_missing_essential_genes == True:
            missing_essential_genes = [gene for gene in np.union1d(top100_essential_genes, top_common_essential_genes) if gene not in top_extracted_genes]
            top_extracted_genes = top_extracted_genes.append(pd.Index(missing_essential_genes))

        existing_columns = depmap_gene_exp_23Q2.columns.intersection(top_extracted_genes)
        
        depmap_gene_exp_23Q2 = depmap_gene_exp_23Q2[existing_columns].sort_index(axis=1)
        depmap_gene_effect_23Q2 = depmap_gene_effect_23Q2[existing_columns].sort_index(axis=1)
        tRCC_gene_exp = tRCC_gene_exp[existing_columns].sort_index(axis=1)
        tRCC_chronos_gene_effects = tRCC_chronos_gene_effects[existing_columns].sort_index(axis=1)

    if mode == "depmap":
        depmap_X_train, depmap_X_test, depmap_Y_train, depmap_Y_test = train_test_split(depmap_gene_exp_23Q2, depmap_gene_effect_23Q2, test_size=test_size, random_state=random_state)

        return [depmap_X_train, depmap_X_test, depmap_Y_train, depmap_Y_test], [tRCC_gene_exp, tRCC_chronos_gene_effects],  {"common_essential_genes": top_common_essential_genes, "top100_essential_genes": top100_essential_genes}
    
    elif mode == "RCC":
        depmap_RCC_data = pd.read_csv("../../datasets/depmap_datasets/CCLIDs/RCC_depmap_data.csv")
        RCC_depmap_gene_exp_23Q2 = depmap_gene_exp_23Q2[depmap_gene_exp_23Q2.index.isin(depmap_RCC_data["depmapId"])]
        RCC_depmap_gene_effect_23Q2 = depmap_gene_effect_23Q2[depmap_gene_effect_23Q2.index.isin(depmap_RCC_data["depmapId"])]

        RCC_X_train, RCC_X_test, RCC_Y_train, RCC_Y_test = train_test_split(RCC_depmap_gene_exp_23Q2, RCC_depmap_gene_effect_23Q2, test_size=test_size, random_state=random_state)

        return [RCC_X_train, RCC_X_test, RCC_Y_train, RCC_Y_test], [tRCC_gene_exp, tRCC_chronos_gene_effects], {"common_essential_genes": top_common_essential_genes, "top100_essential_genes": top100_essential_genes}
    
    elif mode =="ccRCC":
        depmap_ccRCC_data = pd.read_csv('../../datasets/depmap_datasets/CCLIDs/ccRCC_depmap_data.csv')
        ccRCC_depmap_gene_exp_23Q2 = depmap_gene_exp_23Q2[depmap_gene_exp_23Q2.index.isin(depmap_ccRCC_data["depmapId"])]
        ccRCC_depmap_gene_effect_23Q2 = depmap_gene_effect_23Q2[depmap_gene_effect_23Q2.index.isin(depmap_ccRCC_data["depmapId"])]

        ccRCC_X_train, ccRCC_X_test, ccRCC_Y_train, ccRCC_Y_test = train_test_split(ccRCC_depmap_gene_exp_23Q2, ccRCC_depmap_gene_effect_23Q2, test_size=test_size, random_state=random_state)

        return [ccRCC_X_train, ccRCC_X_test, ccRCC_Y_train, ccRCC_Y_test], [tRCC_gene_exp, tRCC_chronos_gene_effects], {"common_essential_genes": top_common_essential_genes, "top100_essential_genes": top100_essential_genes}

def create_model_datasets(X_train, Y_train, X_test, Y_test, GOIs):

    model_X_train = []
    model_Y_train = []

    for gene in GOIs:
        model_X_train.append(X_train.copy())
        model_Y_train.append(Y_train[gene])

    model_X_train = pd.concat(model_X_train).reset_index(drop=True)
    model_Y_train = pd.concat(model_Y_train).reset_index(drop=True)

    model_X_test = []
    model_Y_test = []

    for gene in GOIs:
        model_X_test.append(X_test.copy())
        model_Y_test.append(Y_test[gene])

    model_X_test = pd.concat(model_X_test).reset_index(drop=True)
    model_Y_test = pd.concat(model_Y_test).reset_index(drop=True)

    return model_X_train, model_Y_train, model_X_test, model_Y_test

def restructure_predictions(predictions, GOIs, indexes=None):

    split_predictions = {}
    num_samples = len(predictions) // len(GOIs)
    
    for i, gene in enumerate(GOIs):
        split_predictions[gene] = predictions[i * num_samples:(i + 1) * num_samples]

    restructured_predictions = pd.DataFrame(split_predictions, index= indexes if indexes != None else None)

    return restructured_predictions

import pandas as pd

def restructure_predictions(predictions, GOIs, indexes=None):

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.flatten()

    num_samples_per_gene = len(predictions) // len(GOIs)
    restructured = {}

    for idx, gene in enumerate(GOIs):
        start_idx = idx * num_samples_per_gene
        end_idx = start_idx + num_samples_per_gene
        restructured[gene] = predictions[start_idx:end_idx]

    restructured_df = pd.DataFrame(restructured, index= indexes if indexes is not None else None)

    return restructured_df

def LinePlot(df, 
             line_colors, 
             plot_size=(15, 8), 
             point_size=5, 
             point_alpha=0.7, 
             axes_labels=None, 
             axes_labels_fontsize=10, 
             main_title=None, 
             main_title_fontsize=14, 
             legend_titles=None, 
             xticks_rot=90, 
             xticks_fontsize=6.5):
    
    """
    Plots each column of the dataframe as a line on a graph.

    Args:
    df (pd.DataFrame): DataFrame with genes as index and various scores as columns.
    line_colors (dict): Dictionary mapping column names to colors.
    plot_size (tuple): Size of the plot (width, height).
    point_size (int): Size of the points on each line.
    point_alpha (float): Alpha (transparency) of the points on each line.
    axes_labels (list of str): X and Y-axis labels.
    axes_labels_fontsize (int): Font size for axes labels.
    main_title (str): Main title of the plot.
    main_title_fontsize (int): Font size for the main title.
    legend_titles (dict): Dictionary mapping original column names to legend titles.
    xticks_rot (int): Rotation angle for X-axis ticks.
    xticks_fontsize (int): Font size for X-axis tick labels.
    """

    fig, ax = plt.subplots(figsize=plot_size)

    for column in df.columns:
        color = line_colors.get(column, 'blue')
        ax.plot(df.index, df[column], label=legend_titles.get(column, column) if legend_titles else column, marker='o', markersize=point_size, alpha=point_alpha, color=color)

    if axes_labels:
        ax.set_xlabel(axes_labels[0], fontsize=axes_labels_fontsize)
        ax.set_ylabel(axes_labels[1], fontsize=axes_labels_fontsize)

    if main_title:
        ax.set_title(main_title, fontsize=main_title_fontsize)

    ax.legend()
    ax.grid()

    ax.tick_params(axis='x', labelsize=xticks_fontsize)

    plt.xticks(rotation=xticks_rot)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def GeneRelMiniPlot(rel1_X, rel1_Y, genes,
                    rel2_X=None,
                    rel2_Y=None,
                    models=None,
                    num_cols=6,
                    subplot_size=(2.25, 2.25),
                    point_size=5,
                    alpha=0.5,
                    main_title=None,
                    main_title_fontsize=16,
                    axes_labels=None,
                    axis_label_fontsize=10,
                    axis_tick_fontsize=10,
                    legend_titles=None,
                    sizeProps=[None, None, None, None],
                    wspace=None,
                    hspace=None):
    
    """
    Creates scatter plots for each gene to show the relationship between
    gene expression levels and gene effect scores, with subplot size customization,
    main title, and legend.

    Args:
    rel1_X, rel1_Y (pd.DataFrame): DataFrames with rows as samples and columns as genes.
    genes (list): List of genes to plot.
    rel2_X, rel2_Y (pd.DataFrame): Optional testing DataFrames.
    models (dict): Optional dictionary of LinearRegression models for each gene.
    num_cols (int): Number of columns in the subplot grid.
    subplot_size (tuple): Size of each subplot (width, height).
    point_size (int): Size of the points in the scatter plot.
    main_title (str): Main title of the plot.
    legend_titles (dict): Titles for the legend, keys should match plot labels.
    """

    num_genes = len(genes)
    num_rows = (num_genes + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(subplot_size[0] * num_cols, subplot_size[1] * num_rows))
    
    plt.subplots_adjust(top= sizeProps[0] if sizeProps[0] is not None else None, 
                        bottom= sizeProps[1] if sizeProps[1] is not None else None, 
                        right= sizeProps[2] if sizeProps[2] is not None else None,
                        left= sizeProps[3] if sizeProps[3] is not None else None,
                        wspace=wspace, 
                        hspace=hspace)

    if main_title != None:
        fig.suptitle(main_title, fontsize=main_title_fontsize)

    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

    for i, gene in enumerate(genes):
        ax = axes[i]
        x = rel1_X[gene].values
        y = rel1_Y[gene].values
        ax.scatter(x, y, alpha=alpha, s=point_size, label="rel1")
        ax.set_title(gene)
        ax.grid(True)

        if rel2_X is not None and rel2_Y is not None and gene in rel2_X.columns and gene in rel2_Y.columns:
            x_test = rel2_X[gene].values
            y_test = rel2_Y[gene].values
            ax.scatter(x_test, y_test, alpha=alpha,
                       s=point_size, c="red", label="rel2")

        if models is not None and gene in models:
            model = models[gene]
            x_vals = np.array(ax.get_xlim())
            y_vals = model.intercept_ + model.coef_[0] * x_vals
            ax.plot(x_vals, y_vals, color="black",
                    label="model" if gene == genes[0] else "")

        if axes_labels != None and i == 0:
            ax.set_xlabel(axes_labels[0], fontsize=axis_label_fontsize)
            ax.set_ylabel(axes_labels[1], fontsize=axis_label_fontsize)

        ax.tick_params(axis='both', which='major', labelsize=axis_tick_fontsize)

    for j in range(i + 1, num_rows * num_cols):
        axes[j].set_visible(False)

    if legend_titles is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        updated_labels = [legend_titles.get(label, label) for label in labels]
        fig.legend(handles, updated_labels, loc='lower right')

    plt.show()

def PredGeneRMSE(actual_df, predicted_df, genes=None):
    
    """
    Calculates the RMSE for predicted gene effect scores compared to actual gene effect scores
    across samples for each gene. If no genes are provided, checks that both DataFrames have the same columns.

    Args:
    actual_df (pd.DataFrame): DataFrame with actual gene effect scores.
    predicted_df (pd.DataFrame): DataFrame with predicted gene effect scores.
    genes (list, optional): List of genes to calculate RMSE for. If None, uses all genes in the DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing the RMSE for each gene.
    """

    if genes is None:
        if not actual_df.columns.equals(predicted_df.columns):
            raise ValueError(
                "The columns (genes) of the two DataFrames are not identical.")
        genes = actual_df.columns

    rmse_results = {}

    for gene in genes:
        mse = mean_squared_error(actual_df[gene], predicted_df[gene])
        rmse = np.sqrt(mse)
        rmse_results[gene] = rmse

    return pd.DataFrame.from_dict(rmse_results, orient='index', columns=['RMSE'])