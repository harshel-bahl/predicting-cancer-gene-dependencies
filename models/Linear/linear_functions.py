import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


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


def ExtractGenesNames(columns):
    return [col.split(' ')[0] for col in columns]


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


def GeneLinearity(df1, df2, genes):

    """
    Computes the linearity (Pearson correlation) for each gene in the given list
    of genes, based on the data from two dataframes.

    Args:
    df1 (pd.DataFrame): First DataFrame with rows as samples and columns as genes.
    df2 (pd.DataFrame): Second DataFrame with rows as samples and columns as genes.
    genes (list): List of genes to compute collinearity.

    Returns:
    pd.DataFrame: A DataFrame containing the Pearson correlation for each gene.
    """

    linearity_results = {}

    for gene in genes:
        if gene in df1.columns and gene in df2.columns:
            corr = df1[gene].corr(df2[gene])
            linearity_results[gene] = corr
        else:
            linearity_results[gene] = None

    return pd.DataFrame.from_dict(linearity_results, orient='index', columns=['corr'])


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


def ComputeGeneLinearModels(df1, df2, GOI=None):

    geneModels = {}

    if GOI is not None:
        for gene in GOI:
            X = df1[gene].values.reshape(-1, 1)
            Y = df2[gene]

            model = LinearRegression()
            model.fit(X, Y)

            geneModels[gene] = model

    else:
        for gene in df1.columns:
            X = df1[gene].values.reshape(-1, 1)
            Y = df2[gene]

            model = LinearRegression()
            model.fit(X, Y)

            geneModels[gene] = model

    return geneModels


def ComputeGenePredictions(df1, df2, geneModels):

    genePreds = pd.DataFrame(index=df1.index)

    for gene, model in geneModels.items():

        model = geneModels[gene]

        X_test = df1[gene].values.reshape(-1, 1)
        Y_pred = model.predict(X_test)

        genePreds[gene] = Y_pred

    geneRMSE = PredGeneRMSE(df2[geneModels.keys()], genePreds)

    return genePreds, geneRMSE
