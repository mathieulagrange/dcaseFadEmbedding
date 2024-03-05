import numpy as np
from sklearn import cluster, manifold, metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['figure.figsize'] = [15, 5]
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d

np.random.seed(0)

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        # finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region
                continue
            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


classes = ['Dog Bark', 'Footstep', 'Gunshot', 'Keyboard', 'Moving Motor Vehicle', 'Rain', 'Sneeze/Cough']
classes_fname = ['dog_bark', 'footstep', 'gunshot', 'keyboard', 'moving_motor_vehicle', 'rain', 'sneeze_cough']
embeddings = ['VGGish', 'MS-CLAP', 'PANN-WGM-LOGMEL']
embeddings_fname = ['dcase_vggish', 'dcase_clap-2023', 'dcase_panns-wavegram-logmel']
sur_classes = ['Mixed', 'Impact', 'Impact', 'Impact', 'Texture', 'Texture', 'Mixed']

# Assigning colors based on sur_class
color_dict = {'Impact': 'red', 'Mixed': 'blue', 'Texture': 'green'}

fads = []
for e in embeddings_fname:
    fad = pd.read_excel('./dcase_isomap_data/'+e+'.xlsx')
    fads.append(np.nan_to_num(fad.iloc[:, 1:].to_numpy()))

fig, axes = plt.subplots(ncols=len(embeddings))

for fad_index, fad in enumerate(fads):
    projection = manifold.Isomap(n_components=2).fit_transform(fad)
    sc = axes[fad_index].scatter(projection[:, 0], projection[:, 1], c=range(len(classes)), s=100)
    axes[fad_index].set_title(embeddings[fad_index])

    axes[fad_index].xaxis.set_tick_params(labelbottom=False)
    axes[fad_index].yaxis.set_tick_params(labelleft=False)
    axes[fad_index].set_xticks([])
    axes[fad_index].set_yticks([])

    vor = Voronoi(projection)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    for i, region in enumerate(regions):
        polygon = vertices[region]
        # Only fill the polygons without drawing the edges
        axes[fad_index].fill(*zip(*polygon), color=color_dict[sur_classes[i]], alpha=0.1, edgecolor=None)  # Adjust alpha here

    # Set limits based on scatter plot data
    x_min, x_max = np.min(projection[:, 0]), np.max(projection[:, 0])
    y_min, y_max = np.min(projection[:, 1]), np.max(projection[:, 1])
    axes[fad_index].set_xlim(x_min*1.1, x_max*1.1)
    axes[fad_index].set_ylim(y_min*1.1, y_max*1.1)

legend_elements = []
for cls in ['Impact', 'Mixed', 'Texture']:
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=cls, markerfacecolor=color_dict[cls], markersize=10))

legend_x = 0.1
legend_y = 0.8

legend1 = fig.legend(handles=[legend_elements[0]], bbox_to_anchor=(legend_x, legend_y-0.1), loc='upper left', borderaxespad=0.5, prop={'size': 12})
legend1 = fig.legend(handles=[legend_elements[1]], bbox_to_anchor=(legend_x, legend_y-0.2), loc='upper left', borderaxespad=0.5, prop={'size': 12})
legend1 = fig.legend(handles=[legend_elements[2]], bbox_to_anchor=(legend_x, legend_y-0.5), loc='upper left', borderaxespad=0.5, prop={'size': 12})

order = [1,2,3,0,6,4,5]

legend2 = fig.legend(handles=[sc.legend_elements()[0][1]]+[sc.legend_elements()[0][2]]+[sc.legend_elements()[0][3]], labels=classes, bbox_to_anchor=(legend_x+0.1, legend_y), loc='upper left', borderaxespad=0.5, prop={'size': 12})
legend2 = fig.legend(handles=[sc.legend_elements()[0][0]]+[sc.legend_elements()[0][6]], labels=classes, bbox_to_anchor=(legend_x+0.1, legend_y-0.2), loc='upper left', borderaxespad=0.5, prop={'size': 12})
legend2 = fig.legend(handles=[sc.legend_elements()[0][4]]+[sc.legend_elements()[0][5]], labels=classes, bbox_to_anchor=(legend_x+0.1, legend_y-0.4), loc='upper left', borderaxespad=0.5, prop={'size': 12})

# plt.gca().add_artist(legend1)
plt.tight_layout()
plt.show()
