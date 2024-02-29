import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

# Define a custom function to format float values
def custom_float_format(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value

def get_noisy_correlation_table(perceptual_eval, fad_scores, iterations=100):
    correlation_table_noise = []
    for k in range(iterations):
        #noisy predictions
        std_dev = 1
        noise = np.random.normal(loc=0, scale=std_dev, size=perceptual_eval.shape)
        noisy_perceptual_eval = perceptual_eval + noise
        noisy_perceptual_eval = np.clip(noisy_perceptual_eval, 0, 10)
        correlation, _ = pearsonr(fad_scores, noisy_perceptual_eval)
        correlation_table_noise.append(correlation)
    std_correlation = np.std(correlation_table_noise)
    return(correlation_table_noise, std_correlation)

# Read the Excel files
df_fad_scores = pd.read_excel('./excel_files/fadScores.xlsx', index_col='alg_code', sheet_name=None)
df_perceptual_eval = pd.read_excel('./excel_files/perceptualEval.xlsx', index_col='alg_code', sheet_name=None)
show_p_value = False
# Initialize a dictionary to store correlations
correlation_dict = {}

# Iterate through all sheets of df_perceptual_eval (which corresponds to every perceptual criteria)
for perceptual_eval_sheet_name, df_perceptual_eval_sheet in df_perceptual_eval.items():
    # Skip the sheet if it's empty
    if df_perceptual_eval_sheet.empty:
        continue

    # Sort the DataFrame 
    df_perceptual_eval_sheet = df_perceptual_eval_sheet.sort_index()

    # Drop Baseline row (WARNING: remove this line after we get the dataset for baseline)
    # df_perceptual_eval_sheet = df_perceptual_eval_sheet[df_perceptual_eval_sheet['submission_code'] != 'DCASE2023_baseline_task7']

    # Initialize a dictionary entry for this perceptual criteria
    correlation_dict[perceptual_eval_sheet_name] = {}

    # Iterate through all sheets of df_fad_scores (which corresponds to every type of embedding)
    for fad_scores_sheet_name, df_fad_scores_sheet in df_fad_scores.items():
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

            _, std_correlation_col = get_noisy_correlation_table(df_perceptual_eval_sheet[column], df_fad_scores_sheet[column])
            correlation_table_dict[column+'_std'] = std_correlation_col

        _, std_correlation = get_noisy_correlation_table(perceptual_eval_merged, fad_scores_merged)
        
        # Calculation the correlation of the merged categories array (7x more data points) and add it to the correlation_table_dict
        correlation_merged, p_value_merged = pearsonr(fad_scores_merged, perceptual_eval_merged)

        # Calculate the average correlation of each category and add it to the correlation_table_dict
        avg_category = np.mean(list(correlation_table_dict.values()))
        correlation_table_dict['global'] = correlation_merged
        correlation_table_dict['global_std'] = std_correlation
        p_value_table_dict['global'] = p_value_merged
        correlation_table_dict['avg_category'] = avg_category
        p_value_table_dict['avg_category'] = None

        correlation_dict[perceptual_eval_sheet_name][fad_scores_sheet_name]['correlation'] = correlation_table_dict
        if show_p_value:
            correlation_dict[perceptual_eval_sheet_name][fad_scores_sheet_name]['p_value'] = p_value_table_dict

# Convert correlation_dict to a pandas DataFrame
correlation_df = pd.DataFrame.from_dict({(i, j, k): correlation_dict[i][j][k] 
                                         for i in correlation_dict.keys() 
                                         for j in correlation_dict[i].keys()
                                         for k in correlation_dict[i][j].keys()},
                                        orient='index')

# Reorder the dataframe
correlation_df = correlation_df.stack().unstack(2)
correlation_df = correlation_df.swaplevel().unstack()
correlation_df.reset_index(inplace=True)

if show_p_value:
    # Rename the columns of the multi-index columns to simplify the output
    correlation_df.columns = [f'{col[1]}__{col[0]}' if col[1] else col[0] for col in correlation_df.columns]
else:
    correlation_df = pd.concat([correlation_df.level_0, correlation_df.level_1, correlation_df.correlation], axis=1)

# set criteria and category as indices, and rename them again
correlation_df = correlation_df.set_index(['level_0', 'level_1'])
correlation_df = correlation_df.rename_axis(index={'level_0': 'criteria', 'level_1': 'category'})

# reorder criteria and category
correlation_df = correlation_df.reorder_levels([1, 0], axis=0).sort_index(axis=0, level=[0, 1]).sort_index(axis=1)

# put global_corr and avg_category in first and second row
row_global = correlation_df.xs(key='global', level='category', drop_level=False)
row_global_std = correlation_df.xs(key='global_std', level='category', drop_level=False)
row_avg_category = correlation_df.xs(key='avg_category', level='category', drop_level=False)
correlation_df = correlation_df.drop(index=['global', 'avg_category', 'global_std'], level='category')
correlation_df = pd.concat([row_global, row_global_std, row_avg_category, correlation_df])

formatted_df = correlation_df.map(custom_float_format)

# Save the final_df DataFrame to an Excel file
formatted_df.to_excel('./excel_files/correlation.xlsx')