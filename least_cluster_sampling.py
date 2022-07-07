import os
import argparse
from pathlib import Path
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchmetrics

from datasets.moons import MoonsMaker
from datasets.circles import CirclesMaker
from datasets.gaussian import GaussianMaker
from datasets.blobs import BlobsMaker
from utils.draw import draw_graph
from utils.util import set_random_state
from models.mlp import MLP
from scorers.least_scorer import LeastConfidenceScorer
from scorers.kmeans_scorer import KMeansScorer
from samplers.base_sampler import BaseSampler
from samplers.cluster_sampler import ClusterSampler

def main():
    parser = argparse.ArgumentParser(description='Least confidence with cluster-based sampling')

    parser.add_argument(
        '--data', '-d', required=True, type=str,
        choices=['moons', 'circles', 'gaussian', 'blobs'])
    
    parser.add_argument(
        '--random_state', '-r', default=0, type=int
    )

    args = parser.parse_args()

    ####
    # 乱数設定
    ####
    set_random_state(args.random_state)

    ####
    # データセット/データローダー
    ####
    if args.data == 'moons':
        maker = MoonsMaker(size=1000, random_state=args.random_state)
        num_classes = 2
    elif args.data == 'circles':
        maker = CirclesMaker(size=2000, random_state=args.random_state)
        num_classes = 2
    elif args.data == 'gaussian':
        maker = GaussianMaker(size=1000, random_state=args.random_state)
        num_classes = 2
    elif args.data == 'blobs':
        maker = BlobsMaker(size=1000, random_state=args.random_state)
        num_classes = 4

    train_set = maker.get_train_set()
    valid_set = maker.get_valid_set()
    test_set = maker.get_test_set()
    unlabel_set = maker.get_unlabel_set()
    grid_set = maker.get_grid_set()

    batch_size = 100
    num_workers = 0

    kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
    }

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **kwargs)
    unlabel_loader = DataLoader(unlabel_set, shuffle=False, drop_last=False, **kwargs)
    grid_loader = DataLoader(grid_set, shuffle=False, drop_last=False, **kwargs)

    ####
    # ネットワーク
    ####
    net = MLP(num_classes=num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    print(device)

    ####
    # 損失関数/オプティマイザー/評価関数
    ####
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=1e-1)
    metric = torchmetrics.Accuracy()
    metric.to(device)

    ####
    # スコアラー/サンプラー
    ####
    uncertainty_scorer = LeastConfidenceScorer(net)
    uncertainty_sampler = BaseSampler(uncertainty_scorer, num_samples=2000)

    num_clusters = 4
    diversity_scorer = KMeansScorer(num_clusters, batch_size=batch_size)
    diversity_sampler = ClusterSampler(
        diversity_scorer,
        num_clusters,
        samples_per_cluster=25,
        samples_per_centroid=1,
        samples_per_outlier=1)

    graph_title = 'Least confidence with cluster-based sampling'

    ####
    # 学習
    ####
    max_epoch = 30
    for epoch in range(1, max_epoch+1):

        ####
        # 訓練データ
        ####
        net.train()
        for itteration, (features, labels) in enumerate(train_loader, start=1):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
            optimizer.zero_grad()
            logits = net(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            acc = metric(predictions, labels)

            print(f'epoch: {epoch}/ batch-idx: {itteration}  loss: {loss.item():.6f}  acc: {acc:.4f}')

        acc = metric.compute()
        print(f'epoch: {epoch}  train/acc: {acc:.4f}')
        metric.reset()

        ####
        # テストデータ
        ####
        net.eval()
        for features, labels in test_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.inference_mode():
                logits = net(features)
                predictions = logits.argmax(dim=1)
                acc = metric(predictions, labels)
    
        acc = metric.compute()
        print(f'epoch: {epoch}  test/acc: {acc:.4f}')
        metric.reset()

        ####
        # 不確実性サンプリング
        ####
        for features, _ in unlabel_loader:
            features = features.to(device, non_blocking=True)
            uncertainty_sampler.scoring(features)

        uncertainty_sampling_indices, uncertainty_rest_indices = uncertainty_sampler.sampling()

        uncertainty_set = Subset(unlabel_set, uncertainty_sampling_indices)
        uncertainty_loader = DataLoader(uncertainty_set, shuffle=False, drop_last=False, **kwargs)

        ####
        # 多様性スコアラーの準備
        ####
        for features, _ in uncertainty_loader:
            features = features.to(device, non_blocking=True)
            diversity_scorer.regist_features(features)
        diversity_scorer.post_process()

        ####
        # 多様性サンプリング
        ####
        for features, _ in uncertainty_loader:
            features = features.to(device, non_blocking=True)
            diversity_sampler.scoring(features)

        diversity_sampling_indices, diversity_rest_indices = diversity_sampler.sampling()

        sampling_indices = uncertainty_sampling_indices[diversity_sampling_indices]

        rest_indices = torch.cat([
            uncertainty_rest_indices,
            uncertainty_sampling_indices[diversity_rest_indices]
        ])

        sampled_set = Subset(unlabel_set, sampling_indices)

        ####
        # グリッドデータ
        ####
        net.eval()
        grid_probabilities = []
        uncertainty_scores = []
        diversity_scores = []
        for features, _ in grid_loader:
            features = features.to(device, non_blocking=True)
            with torch.inference_mode():
                logits = net(features)
                probabilities = torch.softmax(logits, dim=1)
                grid_probabilities.append(probabilities.to('cpu'))

            outputs = uncertainty_scorer(features)
            uncertainty_scores.append(outputs['score'].to('cpu'))

            outputs = diversity_scorer(features)
            diversity_scores.append(outputs['score'].to('cpu'))

        grid_probabilities = torch.cat(grid_probabilities)
        uncertainty_scores = torch.cat(uncertainty_scores)
        diversity_scores = torch.cat(diversity_scores)

        ####
        # グラフ描画
        ####
        path = Path(__file__)
        save_dir = path.parent.joinpath('results', args.data, 'least_cluster')
        os.makedirs(save_dir, exist_ok=True)

        draw_graph(
            graph_title + f'/ epoch: {epoch}',
            save_dir.joinpath(f'{args.data}_least_cluster_{epoch:02}.png'),
            train_set,
            sampled_set,
            test_set,
            grid_set,
            grid_probabilities,
            uncertainty_scores,
            diversity_scores)
        
        ####
        # データセット変更
        ####
        train_set = ConcatDataset([train_set, sampled_set])
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **kwargs)

        unlabel_set = Subset(unlabel_set, rest_indices)
        unlabel_loader = DataLoader(unlabel_set, shuffle=False, drop_last=False, **kwargs)        

        ####
        # リセット
        ####
        uncertainty_sampler.reset()
        diversity_scorer.reset()
        diversity_sampler.reset()
        
if __name__ == '__main__':
    main()
