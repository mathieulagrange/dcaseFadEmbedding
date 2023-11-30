import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# Read the Excel files
df_fad_scores = pd.read_excel('fadScores.xlsx', index_col='alg_code', sheet_name=None)
df_perceptual_eval = pd.read_excel('perceptualEval.xlsx', index_col='alg_code', sheet_name=None)

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
    df_perceptual_eval_sheet = df_perceptual_eval_sheet[df_perceptual_eval_sheet['submission_code'] != 'DCASE2023_baseline_task7']

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
        
        # Calculation the correlation of the merged categories array (7x more data points) and add it to the correlation_table_dict
        correlation_merged, p_value_merged = pearsonr(fad_scores_merged, perceptual_eval_merged)
        correlation_table_dict['global_corr'] = correlation_merged
        p_value_table_dict['global_corr'] = p_value_merged

        # Calculate the average correlation of each category and add it to the correlation_table_dict
        avg_category = np.mean(list(correlation_table_dict.values()))
        correlation_table_dict['avg_category'] = avg_category
        p_value_table_dict['avg_category'] = None
        
        correlation_dict[perceptual_eval_sheet_name][fad_scores_sheet_name]['correlation'] = correlation_table_dict
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

# Rename the columns of the multi-index columns to simplify the output
correlation_df.columns = [f'{col[1]}__{col[0]}' if col[1] else col[0] for col in correlation_df.columns]

# set criteria and category as indices, and rename them again
correlation_df = correlation_df.set_index(['level_0', 'level_1'])
correlation_df = correlation_df.rename_axis(index={'level_0': 'criteria', 'level_1': 'category'})

# reorder criteria and category
correlation_df = correlation_df.reorder_levels([1, 0], axis=0).sort_index(axis=0, level=[0, 1]).sort_index(axis=1)

# Save the final_df DataFrame to an Excel file
correlation_df.to_excel('correlation.xlsx')