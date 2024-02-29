import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

fontsize = 14
# Set default font to Times New Roman and default font size to 14
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = fontsize

# Specify the path to your Excel file
file_path = 'excel_files/correlation.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path)
df['category'].ffill(inplace=True)
df.set_index(['category', 'criteria'], inplace=True)
df_transposed = df.T

df_perceptual_eval = pd.read_excel('./excel_files/perceptualEval.xlsx', index_col='alg_code', sheet_name=None)['audio_quality']
categories = df_perceptual_eval.columns[1:]

# Set up the number of rows and columns for subplots
num_categories = len(categories)
num_rows = (num_categories + 1) // 2  # To accommodate odd number of categories
num_cols = 2

fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 45))  # Define the figure and axes

category_rename_dict = {
    'dog_bark' : 'Dog Bark',
    'footstep' : 'Footstep',
    'gunshot' : 'Gunshot',
    'moving_motor_vehicle' : 'Moving Motor Vehicle',
    'rain' : 'Rain',
    'sneeze_cough' : "Sneeze / Cough",
    'keyboard' : 'Keyboard'
}

for idx, category in enumerate(categories, 1):  # Enumerate starting from 1 for subplot numbering
    # Extract relevant data
    global_data = df_transposed.xs(category, level="category", axis=1)
    global_std_data = df_transposed.xs(category+"_std", level="category", axis=1)

    # Select only vggish, clap-2023, and panns-wavegram-logmel
    selected_embeddings = ["vggish", "clap-2023", "panns-wavegram-logmel"]
    global_data = global_data.loc[selected_embeddings]
    global_std_data = global_std_data.loc[selected_embeddings]

    vggish = "VGGish\n(2017)"
    clap_2023 = "MS-CLAP\n(2023)"
    panns_wavegram_logmel = "PANN-WGM\nLogMel (2019)"
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

    # Plot the bars on the current subplot
    ax = axes[(idx - 1) // num_cols, (idx - 1) % num_cols]
    ax.bar(r1, audio_quality_values, width=bar_width, yerr=audio_quality_std, capsize=5, label="Audio Quality", color='#b7e1cd')
    ax.bar(r2, category_fit_values, width=bar_width, yerr=category_fit_std, capsize=5, label="Category Fit", color='#d7b5e8')
    ax.set_title(category_rename_dict[category],fontsize=fontsize, weight='bold')
    # Customize plot
    if idx == 7 or idx == 6:
        ax.set_xticks(list(r1 + bar_width / 2))
        ax.set_xticklabels(system, rotation=80, fontsize=10)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))  # Format y-axis tick labels with 1 decimal place
    ax.set_ylim(-1, 0.6)

    # Add a horizontal line at y=0 with dotted style
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1)

# Adjust spacing between subplots
# plt.subplots_adjust(hspace=1)

# Hide the empty subplot, if exists
if num_categories % 2 != 0:
    empty_subplot_index = num_categories + 1
    axes[(empty_subplot_index - 1) // num_cols, (empty_subplot_index - 1) % num_cols].axis('off')

# Plot the legend outside of the subplots
handles = [plt.Rectangle((0, 0), 1, 1, color='#b7e1cd'), plt.Rectangle((0, 0), 1, 1, color='#d7b5e8')]
labels = ["Audio Quality", "Category Fit"]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.8, 0.15), fancybox=True, shadow=True, fontsize=fontsize)

# Add label to the entire figure
# fig.text(0.1, 0.8, vggish, horizontalalignment='center', fontsize=10, rotation=80)
# fig.text(0.3, 0.8, clap_2023, horizontalalignment='center', fontsize=10, rotation=80)
# fig.text(0.5, 0.8, panns_wavegram_logmel, horizontalalignment='center', fontsize=10, rotation=80)
# plt.tight_layout()
plt.savefig('figures/plot_bar_graph_correlation_subplots.pdf', format='pdf')
plt.show()
