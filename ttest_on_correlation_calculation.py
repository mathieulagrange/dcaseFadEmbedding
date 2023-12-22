import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define a custom function to format float values
def custom_float_format(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value

# Read the Excel files
df_fad_scores = pd.read_excel('./excel_files/fadScores.xlsx', index_col='alg_code', sheet_name=None)
df_perceptual_eval = pd.read_excel('./excel_files/perceptualEval.xlsx', index_col='alg_code', sheet_name=None)
show_p_value = False
show_std = True
# Initialize a dictionary to store correlations
correlation_dict = {}

# Iterate through all sheets of df_fad_scores (which corresponds to every type of embedding)
for fad_scores_sheet_name, df_fad_scores_sheet in df_fad_scores.items():
    # Skip the sheet if it's empty
    if df_fad_scores_sheet.empty:
        continue
    
    df_fad_scores_sheet = df_fad_scores_sheet.sort_index()
    correlation_dict[fad_scores_sheet_name] = {}

    # Iterate through all sheets of df_perceptual_eval (which corresponds to every perceptual criteria)
    for perceptual_eval_sheet_name, df_perceptual_eval_sheet in df_perceptual_eval.items():
        # Skip the sheet if it's empty
        if df_perceptual_eval_sheet.empty:
            continue

        # Sort the DataFrame
        df_perceptual_eval_sheet = df_perceptual_eval_sheet.sort_index()

        # Initialize a dictionary entry for this embedding type
        correlation_dict[fad_scores_sheet_name][perceptual_eval_sheet_name] = {}

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

        correlation_table_noise = []
        for k in range(100):
            #noisy predictions
            std_dev = 1
            noise = np.random.normal(loc=0, scale=std_dev, size=perceptual_eval_merged.shape)
            noisy_perceptual_eval_merged = perceptual_eval_merged + noise
            noisy_perceptual_eval_merged = np.clip(noisy_perceptual_eval_merged, 0, 10)
            correlation, _ = pearsonr(fad_scores_merged, noisy_perceptual_eval_merged)
            correlation_table_noise.append(correlation)
        
        std_correlation = np.std(correlation_table_noise)
        # Calculation the correlation of the merged categories array (7x more data points) and add it to the correlation_table_dict
        correlation_merged, p_value_merged = pearsonr(fad_scores_merged, perceptual_eval_merged)

        # Calculate the average correlation of each category and add it to the correlation_table_dict
        avg_category = np.mean(list(correlation_table_dict.values()))
        correlation_table_dict['global'] = correlation_merged
        correlation_table_dict['global_std'] = std_correlation
        p_value_table_dict['global'] = p_value_merged
        correlation_table_dict['avg_category'] = avg_category
        p_value_table_dict['avg_category'] = None

        correlation_dict[fad_scores_sheet_name][perceptual_eval_sheet_name] = np.array(correlation_table_noise)

for key in correlation_dict.keys():
    correlation_dict[key]['mean_percept'] = (correlation_dict[key]['audio_quality'] +  correlation_dict[key]['category_fit']) / 2

for key in correlation_dict.keys():
    if key != 'panns-wavegram-logmel':
        t_statistic, p_value = stats.ttest_ind(correlation_dict[key]['mean_percept'], correlation_dict['panns-wavegram-logmel']['mean_percept'])
        print(key)
        print(p_value)
