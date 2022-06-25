import torch

class ClusterSampler:
    def __init__(self, scorer, num_clusters, samples_per_cluster, samples_per_centroid, samples_per_outlier):
        self.scorer = scorer
        self.num_clusters = num_clusters
        self.samples_per_cluster = samples_per_cluster
        self.samples_per_centroid = samples_per_centroid
        self.samples_per_outlier = samples_per_outlier

        self.scores = []
        self.cluster_labels = []
    
    def scoring(self, features):
        outputs = self.scorer(features)
        self.scores.append(outputs['score'])
        self.cluster_labels.append(outputs['cluster_label'])
    
    def sampling(self):
        self.scores = torch.cat(self.scores)
        self.cluster_labels = torch.cat(self.cluster_labels)

        sampling_indices = []
        rest_indices = []
        for cluster_label in range(self.num_clusters):
            indices = torch.where(self.cluster_labels == cluster_label)[0]
            sort_args = torch.argsort(self.scores[indices], descending=True)
            sort_indices = indices[sort_args]
            
            centroid_indices = sort_indices[:self.samples_per_centroid]
            sampling_indices.append(centroid_indices)

            random_indices = sort_indices[self.samples_per_centroid:-self.samples_per_outlier]
            random_indices = random_indices[torch.randperm(len(random_indices))]
            samples_per_random = self.samples_per_cluster - self.samples_per_centroid - self.samples_per_outlier
            sampling_indices.append(random_indices[:samples_per_random])
            rest_indices.append(random_indices[samples_per_random:])

            outlier_indices = sort_indices[-self.samples_per_outlier:]
            sampling_indices.append(outlier_indices)

        sampling_indices = torch.cat(sampling_indices)
        rest_indices = torch.cat(rest_indices)

        return sampling_indices, rest_indices

    def reset(self):
        self.scores = []
        self.cluster_labels = []


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parents[1]))

    from torch.utils.data import DataLoader
    from datasets.moons import MoonsMaker
    from scorers.kmeans_scorer import KMeansScorer

    maker = MoonsMaker(1000)
    unlabel_set = maker.get_unlabel_set()
    num_clusters = 4
    batch_size = 100
    unlabel_loader = DataLoader(unlabel_set, batch_size)

    scorer = KMeansScorer(num_clusters=num_clusters, random_state=0, batch_size=batch_size)
    sampler = ClusterSampler(scorer, num_clusters, samples_per_cluster=25, samples_per_centroid=2, samples_per_outlier=2)

    # セントロイドの作成
    for features, _ in unlabel_loader:
        scorer.regist_features(features)
        scorer.update_centroids(features)
    scorer.post_process()

    # クラスタリング
    scores = []
    cluster_labels = []
    for features, _ in unlabel_loader:
        sampler.scoring(features)
        
        outputs = scorer(features)
        scores.append(outputs['score'])
        cluster_labels.append(outputs['cluster_label'])
    scores = torch.cat(scores)
    cluster_labels = torch.cat(cluster_labels)

    sampling_indices, rest_indices = sampler.sampling()
    print(len(sampling_indices))
    print(sampling_indices)

    print(len(rest_indices))
    print(rest_indices)

    print(scores[sampling_indices])
    print(cluster_labels[sampling_indices])
