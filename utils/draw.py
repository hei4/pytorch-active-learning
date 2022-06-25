from cProfile import label
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from matplotlib.patches import Patch
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

def draw_graph(
    graph_title,
    filename,
    train_set,
    sampled_set,
    test_set,
    grid_set,
    grid_probabilities,
    grid_scores):

    train_features = []
    train_labels = []
    for features, labels in train_set:
        train_features.append(features)
        train_labels.append(labels)
    train_features = torch.stack(train_features).numpy()
    train_labels = torch.stack(train_labels).numpy()

    sampled_features = []
    sampled_labels = []
    for features, labels in sampled_set:
        sampled_features.append(features)
        sampled_labels.append(labels)
    sampled_features = torch.stack(sampled_features).numpy()
    sampled_labels = torch.stack(sampled_labels).numpy()
    
    test_features = []
    test_labels = []
    for features, labels in test_set:
        test_features.append(features)
        test_labels.append(labels)
    test_features = torch.stack(test_features).numpy()
    test_labels = torch.stack(test_labels).numpy()

    grid_X = []
    grid_Y = []
    for feature, _ in grid_set:
        grid_X.append(feature[0])
        grid_Y.append(feature[1])
    grid_X = torch.stack(grid_X).reshape(101, 101).numpy()
    grid_Y = torch.stack(grid_Y).reshape(101, 101).numpy()

    xmin = np.min(grid_X)
    xmax = np.max(grid_X)
    ymin = np.min(grid_Y)
    ymax = np.max(grid_X)

    grid_probabilities = grid_probabilities.numpy()

    if grid_probabilities.shape[1] == 2:
        is_contour = True
        cm = ListedColormap(['paleturquoise', 'lightcoral'])
        grid_probabilities = grid_probabilities[:, 0].reshape(101, 101)    
    else:
        num_dim = grid_probabilities.shape[1]
        is_contour = False

        cm = plt.get_cmap('Pastel1')
        
        # colors = np.array([list(cm(0)), list(cm(1)), list(cm(2)), list(cm(3))])
        colors = np.array([list(cm(i)) for i in range(num_dim)])
        colors = colors[:, :3]
        white = np.ones_like(colors, dtype=np.float64)

        grid_probabilities = grid_probabilities[:, :, np.newaxis]
        grid_colors = grid_probabilities * colors + (1. - grid_probabilities) * white
        grid_colors = np.sum(grid_probabilities * grid_colors, axis=1)
        grid_colors = np.clip(grid_colors, 0., 1.)
        grid_colors = grid_colors.reshape(101, 101, 3)        
        
    grid_scores = grid_scores.reshape(101, 101).numpy()

    levels = np.linspace(0, 1, 21)

    fig = plt.figure(1, figsize=(10, 5), facecolor='white')
    fig.suptitle(graph_title, fontsize=16)
    
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 2),
        axes_pad=1.,
        cbar_location='right' if is_contour==True else None,
        cbar_mode='each' if is_contour==True else None,
        cbar_pad=0.1)

    grid[0].set_aspect('equal')
    colorbar = grid[0].contourf(grid_X, grid_Y, grid_scores, cmap='viridis', alpha=0.2)
    grid.cbar_axes[0].colorbar(colorbar)
    if grid_scores.max() - grid_scores.min() < 1e-6:
        grid.cbar_axes[0].tick_params(labelright=False)
    grid[0].scatter(
        train_features[:, 0], train_features[:, 1],
        c=cm(train_labels), alpha=0.5,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    grid[0].scatter(
        sampled_features[:, 0], sampled_features[:, 1],
        c=cm(sampled_labels), alpha=1.0,
        marker='o', s=24, edgecolor='black', linewidth=0.1)
    grid[0].set_xlim(xmin, xmax)
    grid[0].set_ylim(ymin, ymax)
    grid[0].set_xticks(np.linspace(xmin, xmax, 5))
    grid[0].set_yticks(np.linspace(ymin, ymax, 5))
    grid[0].set_title('Training & Sampled Data / Score Distribution', fontsize=10)

    legend_elements = [
        Line2D(
            [0], [0], linestyle='none', marker='o', 
            markeredgewidth=0.1, markeredgecolor='black', markerfacecolor='grey',
            alpha=0.5, markersize=2, label='train'),
        Line2D(
            [0], [0], linestyle='none', marker='o',
            markeredgewidth=0.1, markeredgecolor='black', markerfacecolor='grey',
            alpha=0.5, markersize=6, label='sampled')]
    grid[0].legend(handles=legend_elements)

    grid[1].set_aspect('equal')
    if is_contour == True:
        colorbar = grid[1].contourf(grid_X, grid_Y, grid_probabilities, levels=levels, cmap='RdBu', alpha=0.2)
        grid.cbar_axes[1].colorbar(colorbar)
        grid.cbar_axes[1].set_yticks(np.linspace(0., 1., 5))
    else:
        grid[1].imshow(grid_colors, alpha=0.5, origin='lower', extent=[xmin, xmax, ymin, ymax])

    grid[1].scatter(
        test_features[:, 0], test_features[:, 1],
        c=cm(test_labels), alpha=1.0,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    grid[1].set_xlim(xmin, xmax)
    grid[1].set_ylim(ymin, ymax)
    grid[1].set_xticks(np.linspace(xmin, xmax, 5))
    grid[1].set_yticks(np.linspace(ymin, ymax, 5))
    grid[1].set_title('Test Data / Probability', fontsize=10)

    legend_elements = [
        Line2D(
            [0], [0], linestyle='none', marker='o', 
            markeredgewidth=0.1, markeredgecolor='black', markerfacecolor='grey',
            alpha=0.5, markersize=2, label='test')]
    grid[1].legend(handles=legend_elements)

    path = Path(__file__)
    plt.savefig(path.parent.joinpath(filename))
    plt.close()
    