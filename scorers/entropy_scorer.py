import torch
import math
from model_base_scorer import ModelBaseScorer

class EntropyBasedScorer(ModelBaseScorer):
    def calc_scores_from_logits(self, logits:torch.Tensor):
        probabilities = torch.softmax(logits, dim=1)
        prob_logs = probabilities * torch.log2(probabilities)
        numerator = -torch.sum(prob_logs, dim=1)
        denominator = math.log2(probabilities.size(1))  # torch.log2の引数はTensor限定のためmath.log2

        return numerator / denominator


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parents[1]))

    from models.mlp import MLP

    num_classes = 2
    net = MLP(num_classes=num_classes)
    scorer = EntropyBasedScorer(net)

    features = torch.tensor([
        [-1., -1.],
        [-1., 0.],
        [-1., 1.],
        [0., -1.],
        [0., 0.],
        [0., 1.],
        [1., -1.],
        [1., 0.],
        [1., 1.]], dtype=torch.float32)
    print(f'features: {features}')
    
    outputs = scorer(features)
    scores = outputs['score']
    print(f'scores: {scores}')

    logits = torch.tensor([
        [-1., 0., 1.],
        [0., 1., 2.],
        [0., 1., 10.],
        [0., 10., 100.]], dtype=torch.float32)
    print(f'logits: {logits}')

    scores = scorer.calc_scores_from_logits(logits)
    print(f'scores: {scores}')
