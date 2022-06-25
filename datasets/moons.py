import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset

from base import BaseMaker

class MoonsMaker(BaseMaker):
    def __init__(self, size:int, random_state:int=0) -> None:
        num_classes = 2
        size_per_class = size // num_classes

        np.random.seed(random_state)

        self.noise = 0.1

        self.train_set = self.make_dataset(size_per_class, random_state)
        self.valid_set = self.make_dataset(size_per_class, random_state+1)
        self.test_set = self.make_dataset(size_per_class, random_state+2)
        self.unlabel_set = self.make_dataset(10*size_per_class, random_state+3)

        space = torch.linspace(-2., 2., steps=101)
        grid_x, grid_y = torch.meshgrid(space, space, indexing='ij')
        X_grid = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        self.grid_set = TensorDataset(X_grid, torch.empty(len(X_grid)))

    def make_dataset(self, size_per_class, random_state):
        X, y = make_moons(
            n_samples=(size_per_class, size_per_class),
            noise=self.noise,
            shuffle=True,
            random_state=random_state)
        X[:, 0] -= 0.5
        X[:, 1] -= 0.25
        X = X.astype(np.float32)

        return TensorDataset(torch.tensor(X), torch.tensor(y))

if __name__ == '__main__':
    maker = MoonsMaker(size=1000)

    train_set = maker.get_train_set()
    valid_set = maker.get_valid_set()
    unlabel_set = maker.get_unlabel_set()
    test_set = maker.get_test_set()
    grid_set = maker.get_grid_set()

    print(f'train:   {len(train_set)}')
    print(f'valid:   {len(valid_set)}')
    print(f'test:    {len(test_set)}')
    print(f'unlabel: {len(unlabel_set)}')
    print(f'grid:    {len(grid_set)}')

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
    
    # stack unlabel data
    unlabel_features = []
    unlabel_labels = []
    for feature, label in unlabel_set:
        unlabel_features.append(feature)
        unlabel_labels.append(label)
    unlabel_features = torch.stack(unlabel_features, dim=0)
    unlabel_labels = torch.stack(unlabel_labels)

    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from pathlib import Path

    cm = ListedColormap(['lightblue', 'lightcoral'])

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

    axs[0].set_aspect('equal')
    axs[0].scatter(
        train_features[:, 0], train_features[:, 1],
        c=train_labels, cmap=cm, alpha=0.5,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    axs[0].set_xlim(-2, 2)
    axs[0].set_ylim(-2, 2)

    axs[1].set_aspect('equal')
    axs[1].scatter(
        valid_features[:, 0], valid_features[:, 1],
        c=valid_labels, cmap=cm, alpha=0.5,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    axs[1].set_xlim(-2, 2)
    axs[1].set_ylim(-2, 2)

    axs[2].set_aspect('equal')
    axs[2].scatter(
        test_features[:, 0], test_features[:, 1],
        c=test_labels, cmap=cm, alpha=0.5,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    axs[2].set_xlim(-2, 2)
    axs[2].set_ylim(-2, 2)

    axs[3].set_aspect('equal')
    axs[3].scatter(
        unlabel_features[:, 0], unlabel_features[:, 1],
        c=unlabel_labels, cmap=cm, alpha=0.5,
        marker='o', s=8, edgecolor='black', linewidth=0.1)
    axs[3].set_xlim(-2, 2)
    axs[3].set_ylim(-2, 2)
    
    path = Path(__file__)
    plt.savefig(path.parent.joinpath('check_moons.png'))
    plt.close()