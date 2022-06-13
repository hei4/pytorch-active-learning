from sklearn.cluster import MiniBatchKMeans
import numpy as np
import torch

class KMeansScorer:
    def __init__(self, num_clusters, batch_size, random_state=0) -> None:
        self.num_clusters = num_clusters

        self.features = []
        self.normalize_values = []

        self.kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=batch_size,
            random_state=random_state)
    
    def __call__(self, features):
        distances = torch.clamp_max(self.get_distances(features) / self.normalize_values, 1.)
        cluster_labels = self.get_cluster_labels(features)
        scores = 1. - distances[torch.arange(len(distances)), cluster_labels]
        return {'score': scores, 'cluster_label': cluster_labels}

    def add_features(self, features):
        self.features.append(features)

    def update_centroids(self, features):
        self.kmeans.partial_fit(features.numpy())

    def get_distances(self, features):
        return torch.tensor(self.kmeans.transform(features.numpy()), dtype=torch.float32)
    
    def get_cluster_labels(self, features):
        return torch.tensor(self.kmeans.predict(features.numpy()), dtype=torch.int64)

    def compute_normalize_value(self):
        self.features = torch.cat(self.features)

        distances = self.get_distances(self.features)
        cluster_labels = self.get_cluster_labels(self.features)

        own_distances = distances[torch.arange(len(distances)), cluster_labels]
        for cluster_label in range(self.num_clusters):
            mask = cluster_label == cluster_labels
            normalize_value = own_distances[mask].max()
            self.normalize_values.append(normalize_value)
        self.normalize_values = torch.stack(self.normalize_values)

    def reset(self):
        self.features = []
        self.normalize_values = []
    
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parents[1]))

    from torch.utils.data import DataLoader
    from datasets.moons import MoonsMaker

    maker = MoonsMaker(1000)
    unlabel_set = maker.get_unlabeled_set()
    num_clusters = 4
    batch_size = 100
    unlabel_loader = DataLoader(unlabel_set, batch_size)

    scorer = KMeansScorer(num_clusters=num_clusters, random_state=0, batch_size=batch_size)

    # セントロイドの作成
    for features, _ in unlabel_loader:
        scorer.add_features(features)
        scorer.update_centroids(features)
    scorer.compute_normalize_value()

    # クラスタリング
    scores = []
    cluster_labels = []
    for features, _ in unlabel_loader:
        outputs = scorer(features)
        scores.append(outputs['score'])
        cluster_labels.append(outputs['cluster_label'])
    scores = torch.cat(scores)
    cluster_labels = torch.cat(cluster_labels)

    for cluster_label in range(num_clusters):
        indices = torch.where(cluster_labels == cluster_label)[0]  # クラスターのインデックス
        sort_args = torch.argsort(scores[indices], descending=True)
        sort_indices = indices[sort_args]
        print(len(indices))
        print(indices)
        print(sort_indices)
        print(scores[sort_indices])
        print(cluster_labels[sort_indices], '\n')
