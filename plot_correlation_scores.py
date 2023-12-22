import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

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

ax.bar(r1, audio_quality_values, width=bar_width, yerr=audio_quality_std, capsize=5, label="Audio Quality")
ax.bar(r2, category_fit_values, width=bar_width, yerr=category_fit_std, capsize=5, label="Category Fit")

# Customize plot
plt.xlabel("Embedding")
plt.ylabel("Pearson Correlation")
plt.xticks(list(r1 + bar_width / 2), system, rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('figures/plot_bar_graph_correlation.png')
