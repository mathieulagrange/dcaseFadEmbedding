import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

fontsize = 20

# Specify the path to your Excel file
file_path = 'excel_files/correlation.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path)
df['category'].ffill(inplace=True)
df.set_index(['category', 'criteria'], inplace=True)
df_transposed = df.T

df_perceptual_eval = pd.read_excel('./excel_files/perceptualEval.xlsx', index_col='alg_code', sheet_name=None)['audio_quality']
categories = df_perceptual_eval.columns[1:]


for idx, category in enumerate(categories):
    # Extract relevant data
    global_data = df_transposed.xs(category, level="category", axis=1)
    global_std_data = df_transposed.xs(category+"_std", level="category", axis=1)

    # Select only vggish, clap-2023, and panns-wavegram-logmel
    selected_embeddings = ["vggish", "clap-2023", "panns-wavegram-logmel"]
    global_data = global_data.loc[selected_embeddings]
    global_std_data = global_std_data.loc[selected_embeddings]

    vggish = "VGGish (2017)"
    clap_2023 = "CLAP Microsoft\n(2023)"
    panns_wavegram_logmel = "PANNs-CNN14\nWGM (2019)"
    # Rename labels
    renamed_labels = {
        "vggish": vggish,
        "clap-2023": clap_2023,
        "panns-wavegram-logmel": panns_wavegram_logmel, 
    }

    global_data = global_data.rename(index=renamed_labels)
    global_std_data = global_std_data.rename(index=renamed_labels)

    order = [vggish, panns_wavegram_logmel, clap_2023]
    custom_order = pd.CategoricalDtype(order, ordered=True)
    global_data.index = global_data.index.astype(custom_order)
    global_data = global_data.sort_index()
    global_std_data = global_std_data.loc[global_data.index]

    # Extract criteria names and values
    criteria = global_data.columns
    system = global_data.index.values
    audio_quality_values = global_data["audio_quality"]
    audio_quality_std = global_std_data["audio_quality"]
    category_fit_values = global_data["category_fit"]
    category_fit_std = global_std_data["category_fit"]

    # Set up positions for bars
    bar_width = 0.35
    r1 = np.arange(len(system))
    r2 = [x + bar_width for x in r1]

    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the size (width, height) as needed

    ax.bar(r1, audio_quality_values, width=bar_width, yerr=audio_quality_std, capsize=5, label="Audio Quality", color='#b7e1cd')
    ax.bar(r2, category_fit_values, width=bar_width, yerr=category_fit_std, capsize=5, label="Category Fit", color='#d7b5e8')

    # Customize plot
    plt.xlabel("Embedding", fontsize=fontsize)
    plt.ylabel("Pearson Correlation", fontsize=fontsize)
    plt.xticks(list(r1 + bar_width / 2), system, rotation=80, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)  # Set the font size for y-axis tick labels
    plt.legend(fontsize=fontsize)  # Set the font size for legend
    plt.ylim(-1, 0.6)
    plt.tight_layout()
    plt.savefig('figures/plot_bar_graph_correlation_'+category+'.pdf', format='pdf')
