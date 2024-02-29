import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

fontsize = 20
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

# Extract relevant data
global_data = df_transposed.xs("global", level="category", axis=1)
global_std_data = df_transposed.xs("global_std", level="category", axis=1)

# Select only vggish, clap-2023, and panns-wavegram-logmel
selected_embeddings = ["vggish", "clap-2023", "panns-cnn14-16k", "panns-cnn14-32k", "clap-laion-audio", "clap-laion-music", "panns-wavegram-logmel", "MERT-v1-95M"]
global_data = global_data.loc[selected_embeddings]
global_std_data = global_std_data.loc[selected_embeddings]

vggish = "VGGish\n(2017)"
clap_2023 = "MS-CLAP\n(2023)"
clap_laion_audio = "L-CLAP-audio\n(2022)"
clap_laion_music = "L-CLAP-mus\n(2022)"
panns_cnn14_16k = "PANN-CNN14\n16k (2019)"
panns_cnn14_32k = "PANN-CNN14\n32k (2019)"
panns_wavegram_logmel = "PANN-WGM\nLogMel (2019)"
mert = "MERT-95M\n(2023)"
# Rename labels
renamed_labels = {
    "vggish": vggish,
    "clap-2023": clap_2023,
    "clap-laion-audio": clap_laion_audio,
    "clap-laion-music": clap_laion_music,
    "panns-cnn14-16k": panns_cnn14_16k,
    "panns-cnn14-32k": panns_cnn14_32k,
    "panns-wavegram-logmel": panns_wavegram_logmel, 
    "MERT-v1-95M" : mert
}

global_data = global_data.rename(index=renamed_labels)
global_std_data = global_std_data.rename(index=renamed_labels)

order = [vggish, mert, panns_cnn14_16k, panns_cnn14_32k, panns_wavegram_logmel, clap_laion_music, clap_laion_audio, clap_2023]
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

# Add a horizontal line at y=0 with dotted style
ax.axhline(y=0, color='black', linestyle=':', linewidth=1)

# Customize plot
# plt.xlabel("Embedding", fontsize=fontsize*1.3)
plt.ylabel("Pearson Correlation", fontsize=fontsize*1.3)
plt.xticks(list(r1 + bar_width / 2), system, rotation=80, fontsize=fontsize)
plt.yticks(fontsize=fontsize)  # Set the font size for y-axis tick labels
plt.legend(fontsize=fontsize)  # Set the font size for legend
plt.tight_layout()
plt.savefig('figures/plot_bar_graph_correlation.pdf', format='pdf')
