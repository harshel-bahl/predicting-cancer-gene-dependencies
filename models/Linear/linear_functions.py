import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stats
import itertools


def check_df_similarity(dfs, check='both'):

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


def intersect_df(dfs, match="both"):

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


def extract_gene_names(columns):
    return [col.split(' ')[0] for col in columns]


def select_features(X_train, Y_train_col, quartiles=10, top_n=20, alpha=0.05):

    X_medians = X_train.median()
    Y_quantiles = pd.qcut(Y_train_col, quartiles, labels=False)

    X_binary = X_train.gt(X_medians, axis='columns')

    pearson_results = {}
    spearman_results = {}
    chi2_results = {}

    for feature in X_train.columns:
        feature_data = X_train[feature]

        if np.std(feature_data) == 0 or np.std(Y_train_col) == 0:
            continue

        pearson_corr, pearson_p = stats.pearsonr(feature_data, Y_train_col)
        spearman_corr, spearman_p = stats.spearmanr(feature_data, Y_train_col)

        if pearson_p < alpha:
            pearson_results[feature] = (abs(pearson_corr), pearson_p)
        if spearman_p < alpha:
            spearman_results[feature] = (abs(spearman_corr), spearman_p)

        contingency_table = pd.crosstab(X_binary[feature], Y_quantiles)
        chi2, p = stats.chi2_contingency(contingency_table)[:2]

        if p < alpha:
            chi2_results[feature] = (chi2, p)

    selected_features = set()
    for result_set in [pearson_results, spearman_results, chi2_results]:
        for feature, _ in sorted(result_set.items(), key=lambda x: x[1][0], reverse=True)[:top_n]:
            selected_features.add(feature)

    return list(selected_features)


def train_mLinear_model(X_train, Y_train, features, add_constant=True):

    X = X_train[features]

    model = sm.OLS(Y_train, sm.add_constant(
        X) if add_constant == True else X).fit()

    return model


def predict_mLinear_model(X_test, Y_test, model, features, add_constant=True):

    X = X_test[features]

    Y_pred = model.predict(sm.add_constant(X) if add_constant == True else X)

    pearsonCorr = calc_pearson_corr(Y_test, Y_pred)
    RMSE = calc_RMSE(Y_test, Y_pred)
    MAE = calc_MAE(Y_test, Y_pred)
    RSquared = calc_RSquared(Y_test, Y_pred)

    return Y_test, Y_pred, pearsonCorr, RMSE, MAE, RSquared

def calc_pearson_corr(actual, predicted, columns=None):

    if isinstance(actual, pd.DataFrame) and isinstance(predicted, pd.DataFrame):
        if columns is None:
            if not actual.columns.equals(predicted.columns):
                raise ValueError("The columns of the two DataFrames are not identical.")
            columns = actual.columns

        correlation_results = {}
        for column in columns:
            corr, p_value = stats.pearsonr(actual[column], predicted[column])
            correlation_results[column] = {'corr': corr, 'p-value': p_value}

        return pd.DataFrame.from_dict(correlation_results, orient='index')

    elif isinstance(actual, pd.Series) and isinstance(predicted, pd.Series):
        if actual.index.equals(predicted.index):
            corr, p_value = stats.pearsonr(actual, predicted)
            return {'corr': corr, 'p-value': p_value}
        else:
            raise ValueError("The indices of the two Series are not identical.")

    else:
        raise TypeError("Inputs must both be pandas DataFrames or pandas Series.")


def calc_RMSE(actual, predicted, columns=None):

    if isinstance(actual, pd.DataFrame) and isinstance(predicted, pd.DataFrame):
        if columns is None:
            if not actual.columns.equals(predicted.columns):
                raise ValueError(
                    "The columns of the two DataFrames are not identical.")
            columns = actual.columns

        rmse_results = {}
        for column in columns:
            mse = mean_squared_error(actual[column], predicted[column])
            rmse_results[column] = np.sqrt(mse)

        return pd.DataFrame.from_dict(rmse_results, orient='index', columns=['RMSE'])

    elif isinstance(actual, pd.Series) and isinstance(predicted, pd.Series):
        if actual.index.equals(predicted.index):
            return np.sqrt(mean_squared_error(actual, predicted))
        else:
            raise ValueError(
                "The indices of the two Series are not identical.")

    else:
        raise TypeError(
            "Inputs must both be pandas DataFrames or pandas Series.")


def calc_MAE(actual, predicted, columns=None):

    if isinstance(actual, pd.DataFrame) and isinstance(predicted, pd.DataFrame):
        if columns is None:
            if not actual.columns.equals(predicted.columns):
                raise ValueError(
                    "The columns of the two DataFrames are not identical.")
            columns = actual.columns

        mae_results = {}
        for column in columns:
            mae = mean_absolute_error(actual[column], predicted[column])
            mae_results[column] = mae

        return pd.DataFrame.from_dict(mae_results, orient='index', columns=['MAE'])

    elif isinstance(actual, pd.Series) and isinstance(predicted, pd.Series):
        if actual.index.equals(predicted.index):
            return mean_absolute_error(actual, predicted)
        else:
            raise ValueError(
                "The indices of the two Series are not identical.")

    else:
        raise TypeError(
            "Inputs must both be pandas DataFrames or pandas Series.")


def calc_RSquared(actual, predicted, columns=None):

    if isinstance(actual, pd.DataFrame) and isinstance(predicted, pd.DataFrame):
        if columns is None:
            if not actual.columns.equals(predicted.columns):
                raise ValueError(
                    "The columns of the two DataFrames are not identical.")
            columns = actual.columns

        r_squared_results = {}
        for column in columns:
            r2 = r2_score(actual[column], predicted[column])
            r_squared_results[column] = r2

        return pd.DataFrame.from_dict(r_squared_results, orient='index', columns=['R-Squared'])

    elif isinstance(actual, pd.Series) and isinstance(predicted, pd.Series):
        if actual.index.equals(predicted.index):
            return r2_score(actual, predicted)
        else:
            raise ValueError(
                "The indices of the two Series are not identical.")

    else:
        raise TypeError(
            "Inputs must both be pandas DataFrames or pandas Series.")


def predictions_miniplot(Y_act1, Y_pred1, genes,
                         Y_act2=None, 
                         Y_pred2=None,
                         num_cols=6,
                         subplot_size=(2.25, 2.25),
                         point_size=5,
                         alpha_dict=None,
                         main_title=None,
                         main_title_fontsize=16,
                         axes_labels=None,
                         axis_label_fontsize=10,
                         axis_tick_fontsize=10,
                         legend_titles=None,
                         sizeProps=[None, None, None, None],
                         wspace=None,
                         hspace=None,
                         equal_axes_scale=False,
                         line_width=1):

    num_genes = len(genes)
    num_rows = (num_genes + num_cols - 1) // num_cols

    if equal_axes_scale:
        all_values = np.concatenate([Y_act1.values.flatten(), Y_pred1.values.flatten(), 
                                    Y_act2.values.flatten() if Y_act2 is not None else [], 
                                    Y_pred2.values.flatten() if Y_pred2 is not None else []])
        global_min, global_max = all_values.min(), all_values.max()

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(
        subplot_size[0] * num_cols, subplot_size[1] * num_rows))

    plt.subplots_adjust(top=sizeProps[0] if sizeProps[0] is not None else None,
                        bottom=sizeProps[1] if sizeProps[1] is not None else None,
                        right=sizeProps[2] if sizeProps[2] is not None else None,
                        left=sizeProps[3] if sizeProps[3] is not None else None,
                        wspace=wspace,
                        hspace=hspace)

    if main_title is not None:
        fig.suptitle(main_title, fontsize=main_title_fontsize)

    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

    for i, gene in enumerate(genes):
        ax = axes[i]

        alpha_rel1 = alpha_dict['rel1'] if alpha_dict and 'rel1' in alpha_dict else 1
        alpha_rel2 = alpha_dict['rel2'] if alpha_dict and 'rel2' in alpha_dict else 1

        ax.scatter(Y_act1[gene].values, Y_pred1[gene].values, alpha=alpha_rel1, s=point_size, label="rel1")
        
        if Y_act2 is not None and Y_pred2 is not None:
            ax.scatter(Y_act2[gene].values, Y_pred2[gene].values, alpha=alpha_rel2, s=point_size, label="rel2")

        min_val = min(Y_act1[gene].values.min(), Y_act2[gene].values.min()) if Y_act2 is not None else Y_act1[gene].values.min()
        max_val = max(Y_act1[gene].values.max(), Y_act2[gene].values.max()) if Y_act2 is not None else Y_act1[gene].values.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=line_width)

        ax.set_title(gene)
        ax.grid(True)

        if equal_axes_scale:
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)

        if axes_labels is not None and i == 0:
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


def bar_chart_multi_col(df,
                        columns_to_plot,
                        plot_size=(15, 8),
                        bar_width=0.75, 
                        space_between_groups=0.1,
                        left_space=1.5,
                        right_space=1.5,
                        axes_labels=None,
                        axes_labels_fontsize=10,
                        main_title=None,
                        main_title_fontsize=14,
                        xticks_rot=90,
                        xticks_fontsize=6.5):

    fig, ax = plt.subplots(figsize=plot_size)
    n_cols = len(columns_to_plot)
    indices = np.arange(len(df)) * (bar_width * n_cols + space_between_groups)

    for i, column in enumerate(columns_to_plot):
        ax.bar(indices + i * bar_width, df[column], bar_width, label=column)

    ax.set_xlabel(axes_labels[0], fontsize=axes_labels_fontsize) if axes_labels else None
    ax.set_ylabel(axes_labels[1], fontsize=axes_labels_fontsize) if axes_labels else None
    ax.set_title(main_title, fontsize=main_title_fontsize) if main_title else None
    ax.set_xticks(indices + bar_width * (n_cols - 1) / 2)
    ax.set_xticklabels(df.index, rotation=xticks_rot, fontsize=xticks_fontsize)
    ax.legend()
    ax.grid()

    left_limit = indices[0] - left_space
    right_limit = indices[-1] + bar_width * n_cols + right_space
    ax.set_xlim(left_limit, right_limit)

    plt.tight_layout()
    plt.show()


def bar_chart_sing_col(df, column,
                       special_colors=None, 
                       default_color='grey',  
                       legend_labels=None,
                       plot_size=(15, 8),
                       bar_width=0.4,
                       left_space=1.5,  
                       right_space=1.5, 
                       axes_labels=None,
                       axes_labels_fontsize=10,
                       main_title=None,
                       main_title_fontsize=14,
                       xticks_rot=90,
                       xticks_fontsize=6.5):

    fig, ax = plt.subplots(figsize=plot_size)
    indices = np.arange(len(df))

    for i, index in enumerate(df.index):
        color = special_colors.get(index, default_color) if special_colors else default_color
        ax.bar(indices[i], df.loc[index, column], bar_width, color=color)

    ax.set_xlabel(axes_labels[0], fontsize=axes_labels_fontsize) if axes_labels else None
    ax.set_ylabel(axes_labels[1], fontsize=axes_labels_fontsize) if axes_labels else None
    ax.set_title(main_title, fontsize=main_title_fontsize) if main_title else None
    ax.set_xticks(indices)
    ax.set_xticklabels(df.index, rotation=xticks_rot, fontsize=xticks_fontsize)

    if legend_labels:
        handles = [plt.Rectangle((0,0),1,1, color=color) for color in legend_labels.values()]
        labels = legend_labels.keys()
        ax.legend(handles, labels)

    ax.grid()

    left_limit = indices[0] - left_space
    right_limit = indices[-1] + bar_width + right_space
    ax.set_xlim(left_limit, right_limit)

    plt.tight_layout()
    plt.show()


def calc_corr_freq(df, column_name):
    
    intervals = np.linspace(0, 1, 11)
    interval_labels = [f"{intervals[i]:.1f}-{intervals[i+1]:.1f}" for i in range(10)]

    frequency = {label: 0 for label in interval_labels}

    for value in df[column_name]:
        for i in range(10):
            if intervals[i] <= value < intervals[i+1]:
                interval_key = interval_labels[i]
                frequency[interval_key] += 1
                break

    frequency_table = pd.DataFrame(list(frequency.items()), columns=['Deciles', 'Frequency'])
    
    return frequency_table





def line_plot(df,
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
        ax.plot(df.index, df[column], label=legend_titles.get(
            column, column) if legend_titles else column, marker='o', markersize=point_size, alpha=point_alpha, color=color)

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

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(
        subplot_size[0] * num_cols, subplot_size[1] * num_rows))

    plt.subplots_adjust(top=sizeProps[0] if sizeProps[0] is not None else None,
                        bottom=sizeProps[1] if sizeProps[1] is not None else None,
                        right=sizeProps[2] if sizeProps[2] is not None else None,
                        left=sizeProps[3] if sizeProps[3] is not None else None,
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

        ax.tick_params(axis='both', which='major',
                       labelsize=axis_tick_fontsize)

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
