import numpy as np
import math
from sklearn.datasets import make_gaussian_quantiles
import torch
from torch.utils.data import TensorDataset

from base import BaseMaker

class GaussianMaker(BaseMaker):
    def __init__(self, size:int, random_state:int=0) -> None:
        num_classes = 2
        size_per_class = size // num_classes

        np.random.seed(random_state)

        self.train_set = self.make_dataset(size_per_class, random_state)
        self.valid_set = self.make_dataset(size_per_class, random_state+1)
        self.test_set = self.make_dataset(size_per_class, random_state+2)
        self.unlabeled_set = self.make_dataset(10*size_per_class, random_state+3)

        space = torch.linspace(-4., 4., steps=101)
        grid_x, grid_y = torch.meshgrid(space, space, indexing='ij')
        X_grid = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        self.grid_set = TensorDataset(X_grid, torch.empty(len(X_grid)))
    
    def make_dataset(self, size_per_class, random_state):
        X, y = make_gaussian_quantiles(
            n_samples=2*size_per_class,
            n_classes=2,
            shuffle=True,
            random_state=random_state)
        X = X.astype(np.float32)
        
        return TensorDataset(torch.tensor(X), torch.tensor(y))

if __name__ == '__main__':
    maker = GaussianMaker(size=1000)

    train_set = maker.get_train_set()
    valid_set = maker.get_valid_set()
    test_set = maker.get_test_set()
    unlabeled_set = maker.get_unlabeled_set()
    grid_set = maker.get_grid_set()

    print(f'train:     {len(train_set)}')
    print(f'valid:     {len(valid_set)}')
    print(f'test:      {len(test_set)}')
    print(f'unlabeled: {len(unlabeled_set)}')
    print(f'grid:      {len(grid_set)}')
    
    feature, label = train_set[0]
    print(f'feature... {feature.shape} {feature.dtype}')
    print(f'label...   {label.shape} {label.dtype}')

    # stack train data
    train_features = []
    train_labels = []
    for feature, label in train_set:
        train_features.append(feature)
        train_labels.append(label)
    train_features = torch.stack(train_features, dim=0)
    train_labels = torch.stack(train_labels)

    # stack valid data
    valid_features = []
    valid_labels = []
    for feature, label in valid_set:
        valid_features.append(feature)
        valid_labels.append(label)
    valid_features = torch.stack(valid_features, dim=0)
    valid_labels = torch.stack(valid_labels)

    # stack test data
    test_features = []
    test_labels = []
    for feature, label in test_set:
        test_features.append(feature)
        test_labels.append(label)
    test_features = torch.stack(test_features, dim=0)
    test_labels = torch.stack(test_labels)

    # stack unlabeled data
    unlabeled_features = []
    unlabeled_labels = []
    for feature, label in unlabeled_set:
        unlabeled_features.append(feature)
        unlabeled_labels.append(label)
    unlabeled_features = torch.stack(unlabeled_features, dim=0)
    unlabeled_labels = torch.stack(unlabeled_labels)

    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from pathlib import Path

    pastel = plt.get_cmap('Pastel1')
    cm = ListedColormap([pastel(0), pastel(1), pastel(2)])

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

    axs[0].set_aspect('equal')
    axs[0].scatter(
        train_features[:, 0], train_features[:, 1],
        c=train_labels, cmap=cm, alpha=0.5,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    axs[0].set_xlim(-4, 4)
    axs[0].set_ylim(-4, 4)

    axs[1].set_aspect('equal')
    axs[1].scatter(
        valid_features[:, 0], valid_features[:, 1],
        c=valid_labels, cmap=cm, alpha=0.5,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    axs[1].set_xlim(-4, 4)
    axs[1].set_ylim(-4, 4)

    axs[2].set_aspect('equal')
    axs[2].scatter(
        test_features[:, 0], test_features[:, 1],
        c=test_labels, cmap=cm, alpha=0.5,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    axs[2].set_xlim(-4, 4)
    axs[2].set_ylim(-4, 4)
    
    axs[3].set_aspect('equal')
    axs[3].scatter(
        unlabeled_features[:, 0], unlabeled_features[:, 1],
        c=unlabeled_labels, cmap=cm, alpha=0.5,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    axs[3].set_xlim(-4, 4)
    axs[3].set_ylim(-4, 4)

    path = Path(__file__)
    plt.savefig(path.parent.joinpath('check_gaussian.png'))
    plt.close()