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
from models.mlp import MLP
from scorers.least_scorer import LeastConfidenceScorer
from scorers.margin_scorer import MarginConfidenceScorer
from scorers.ratio_scorer import RatioConfidenceScorer
from scorers.entropy_scorer import EntropyConfidenceScorer
from samplers.base_sampler import BaseSampler

def main():
    parser = argparse.ArgumentParser(description='Uncertainty Sampling')

    parser.add_argument(
        '--algorithm', '-a', required=True, type=str,
        choices=['least', 'margin', 'ratio', 'entropy'])
    
    parser.add_argument(
        '--data', '-d', required=True, type=str,
        choices=['moons', 'circles', 'gaussian', 'blobs'])
    
    args = parser.parse_args()

    ####
    # データセット/データローダー
    ####

    if args.data == 'moons':
        maker = MoonsMaker(size=1000)
        num_classes = 2
    elif args.data == 'circles':
        maker = CirclesMaker(size=2000)
        num_classes = 2
    elif args.data == 'gaussian':
        maker = GaussianMaker(size=1000)
        num_classes = 2
    elif args.data == 'blobs':
        maker = BlobsMaker(size=1000)
        num_classes = 4

    train_set = maker.get_train_set()
    test_set = maker.get_test_set()
    unlabeled_set = maker.get_unlabeled_set()
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
    unlabeled_loader = DataLoader(unlabeled_set, shuffle=False, drop_last=False, **kwargs)
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

    if args.algorithm == 'least':
        scorer = LeastConfidenceScorer()
        graph_title = 'Least Confidence Sampling'
    elif args.algorithm == 'margin':
        scorer = MarginConfidenceScorer()
        graph_title = 'Margin Confidence Sampling'
    elif args.algorithm == 'ratio':
        scorer = RatioConfidenceScorer()
        graph_title = 'Ratio Confidence Sampling'
    elif args.algorithm == 'entropy':
        scorer = EntropyConfidenceScorer()
        graph_title = 'Entropy Sampling'

    sampler = BaseSampler(scorer, num_samples=100)
    
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
        # ラベルなしデータ
        ####
        net.eval()
        for features, _ in unlabeled_loader:
            features = features.to(device, non_blocking=True)

            with torch.inference_mode():
                logits = net(features)
                sampler.scoring(logits)

        sampling_indices, rest_indices = sampler.sampling()
        sampler.reset()

        sampled_set = Subset(unlabeled_set, sampling_indices)

        ####
        # グリッドデータ
        ####
        net.eval()
        grid_probabilities = []
        grid_scores = []
        for features, _ in grid_loader:
            features = features.to(device, non_blocking=True)
            with torch.inference_mode():
                logits = net(features)
                probabilities = torch.softmax(logits, dim=1)
                grid_probabilities.append(probabilities.to('cpu'))
                
                outputs = scorer(logits)
                scores = outputs['score']
                grid_scores.append(scores.to('cpu'))
        grid_probabilities = torch.cat(grid_probabilities)
        grid_scores = torch.cat(grid_scores)

        ####
        # グラフ描画
        ####
        path = Path(__file__)
        save_dir = path.parent.joinpath('results', args.data, args.algorithm)
        os.makedirs(save_dir, exist_ok=True)

        draw_graph(
            graph_title + f'/ epoch: {epoch}',
            save_dir.joinpath(f'{args.data}_{args.algorithm}_{epoch:02}.png'),
            train_set,
            sampled_set,
            test_set,
            grid_set,
            grid_probabilities,
            grid_scores)
        
        ####
        # データセット変更
        ####
        train_set = ConcatDataset([train_set, sampled_set])
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **kwargs)

        unlabeled_set = Subset(unlabeled_set, rest_indices)
        unlabeled_loader = DataLoader(unlabeled_set, shuffle=False, drop_last=False, **kwargs)        
        
if __name__ == '__main__':
    main()
