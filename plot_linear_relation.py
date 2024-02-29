import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker

# Define a custom function to format float values
def custom_float_format(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value

embeddings_to_filter = ['vggish', 'clap-2023', 'panns-wavegram-logmel']
embedding_rename_dict = {
    'vggish' : "VGGish\n(2017)",
    'clap-2023' : "CLAP Microsoft\n(2023)",
    'panns-wavegram-logmel' : "PANNs-CNN14\nWGM (2019)"
}
perc_rename_dict = {
    'audio_quality' : 'Audio Quality',
    'category_fit' : 'Category Fit'
}

fontsize = 23
# Set default font and font size
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = fontsize

# Read the Excel files
df_fad_scores = pd.read_excel('./excel_files/fadScores.xlsx', index_col='alg_code', sheet_name=None)
df_perceptual_eval = pd.read_excel('./excel_files/perceptualEval.xlsx', index_col='alg_code', sheet_name=None)
show_p_value = False
# Initialize a dictionary to store correlations
correlation_dict = {}

# Create a subplot grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Iterate through all sheets of df_perceptual_eval (which corresponds to every perceptual criteria)
for idx_perc, (perceptual_eval_sheet_name, df_perceptual_eval_sheet) in enumerate(df_perceptual_eval.items()):
    # Skip the sheet if it's empty
    if df_perceptual_eval_sheet.empty:
        continue

    # Sort the DataFrame 
    df_perceptual_eval_sheet = df_perceptual_eval_sheet.sort_index()

    # Drop Baseline row (WARNING: remove this line after we get the dataset for baseline)
    # df_perceptual_eval_sheet = df_perceptual_eval_sheet[df_perceptual_eval_sheet['submission_code'] != 'DCASE2023_baseline_task7']

    # Initialize a dictionary entry for this perceptual criteria
    correlation_dict[perceptual_eval_sheet_name] = {}

    df_fad_scores = {key: df_fad_scores[key] for key in embeddings_to_filter if key in df_fad_scores}

    # Iterate through all sheets of df_fad_scores (which corresponds to every type of embedding)
    for idx_cat, (fad_scores_sheet_name, df_fad_scores_sheet) in enumerate(df_fad_scores.items()):
        # Skip the sheet if it's empty
        if df_fad_scores_sheet.empty:
            continue

        # Sort the DataFrame
        df_fad_scores_sheet = df_fad_scores_sheet.sort_index()

        # Initialize a dictionary entry for this embedding type
        correlation_dict[perceptual_eval_sheet_name][fad_scores_sheet_name] = {}

        # Calculate Pearson correlation for each column
        correlation_table_dict = {}
        p_value_table_dict = {}


        # perceptual_eval_merged and fad_scores_merged are used to calculate global correlation
        perceptual_eval_merged = np.array([])
        fad_scores_merged = np.array([])
        for column in df_perceptual_eval_sheet.columns[1:]:
            correlation, p_value = pearsonr(df_fad_scores_sheet[column], df_perceptual_eval_sheet[column])
            correlation_table_dict[column] = correlation
            p_value_table_dict[column] = p_value
            perceptual_eval_merged = np.concatenate((perceptual_eval_merged, df_perceptual_eval_sheet[column]))
            fad_scores_merged = np.concatenate((fad_scores_merged, df_fad_scores_sheet[column]))


        # Plot on the corresponding subplot
        ax = axes[idx_perc, idx_cat]
        if idx_perc == 1 & idx_cat == 1:
            ax.set_xlabel('Perceptual Evaluation', fontsize=fontsize)        
        if idx_cat == 0:
            ax.set_ylabel(perc_rename_dict[perceptual_eval_sheet_name], fontsize=fontsize, labelpad=45, weight='bold')
            # ax.yaxis.set_label_coords(-0.1, -1)
        if idx_perc == 0:
            ax.set_title(embedding_rename_dict[fad_scores_sheet_name], weight='bold')
            ax.title.set_size(fontsize)
        ax.scatter(perceptual_eval_merged, fad_scores_merged, alpha=0.5)
        ax.set_xlim(0, 10)  # Set x-axis limit to 1
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))  # Show ticks every 0.5 units on x-axis

fig.text(0.08, 0.5, 'FAD Scores', ha='center', va='center', rotation=90, fontsize=fontsize)
fig.savefig('figures/plot_linear_relations.pdf', bbox_inches='tight')