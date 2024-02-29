import numpy as np
from sklearn import cluster, manifold, metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['figure.figsize'] = [15, 5]
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
import pandas as pd

np.random.seed(0)

classes = ['Dog Bark', 'Footstep', 'Gunshot', 'Keyboard', 'Moving Motor Vehicle', 'Rain', 'Sneeze/Cough']
classes_fname = ['dog_bark', 'footstep', 'gunshot', 'keyboard', 'moving_motor_vehicle', 'rain', 'sneeze_cough']
embeddings = ['VGGish', 'MS-CLAP', 'PANN-WGM-LOGMEL']
embeddings_fname = ['dcase_vggish', 'dcase_clap-2023', 'dcase_panns-wavegram-logmel']

fads = []
for e in embeddings_fname:
    fad = pd.read_excel('./dcase_isomap_data/'+e+'.xlsx')
    fads.append(np.nan_to_num(fad.iloc[:, 1:].to_numpy()))

fig, axes = plt.subplots(ncols=len(embeddings))

#standardized color map
cmap='tab10'
#custom color map
# colors = ['#000080', '#4169E1', '#1E90FF', '#CD5C5C', '#DC143C', '#B22222', '#87CEEB']
# cmap=ListedColormap(colors)

for fad_index, fad in enumerate(fads):
    projection = manifold.Isomap(n_components=2).fit_transform(fad)
    # Using 
    # sc = axes[fad_index].scatter(projection[:, 0], projection[:, 1], c=range(len(classes)), cmap=ListedColormap(colors), s=100)
    sc = axes[fad_index].scatter(projection[:, 0], projection[:, 1], c=range(len(classes)), cmap=cmap, s=100)
    axes[fad_index].set_title(embeddings[fad_index])

    # Hide X and Y axes label marks
    axes[fad_index].xaxis.set_tick_params(labelbottom=False)
    axes[fad_index].yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    axes[fad_index].set_xticks([])
    axes[fad_index].set_yticks([])

fig.legend(handles=sc.legend_elements()[0], labels=classes, bbox_to_anchor=(0.2, 0.8), loc='upper left', borderaxespad=0.5, prop={'size': 12})
plt.tight_layout()

plt.savefig('figures/plot_dcase_isomap.pdf')