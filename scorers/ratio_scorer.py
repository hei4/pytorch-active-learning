import torch
from model_base_scorer import ModelBaseScorer

class RatioConfidenceScorer(ModelBaseScorer):
    def calc_scores_from_logits(self, logits:torch.Tensor):
        probabilities = torch.softmax(logits, dim=1)
        topk_prob = torch.topk(probabilities, k=2, dim=1).values
        ratio_prob = topk_prob[:, 1] / topk_prob[:, 0]

        return ratio_prob

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parents[1]))

    from models.mlp import MLP

    num_classes = 2
    net = MLP(num_classes=num_classes)
    scorer = RatioConfidenceScorer(net)

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
