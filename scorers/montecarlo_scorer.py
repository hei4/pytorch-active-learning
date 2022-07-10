import torch
from torch import nn
from models.dropout_mlp import DropoutMLP

class MontecarloScorer:
    def __init__(self, net, num_repeats=50) -> None:        
        self.net = DropoutMLP(net)
        self.num_repeats = num_repeats

    def __call__(self, features:torch.Tensor):
        self.net.train()    # モンテカルロ・ドロップアウトのためtrainモード
        repeat_logits = []
        with torch.inference_mode():    # 逆伝播しないためinferenceモード
            for _ in range(self.num_repeats):
                logits = self.net(features)
                repeat_logits.append(logits)
        repeat_logits = torch.stack(repeat_logits, dim=-1)
        scores = self.calc_scores_from_repeat_logits(repeat_logits)
        return {'score': scores}
    
    def calc_scores_from_repeat_logits(self, repeat_logits):
        repeat_probabilities = torch.softmax(repeat_logits, dim=1)
        variance = torch.var(repeat_probabilities, dim=-1)
        return variance.mean(dim=-1)

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parents[1]))

    from models.mlp import MLP

    num_classes = 4
    net = MLP(num_classes=num_classes)
    scorer = MontecarloScorer(net)

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
        [0., 1., 2.],
        [0., 1., 2.],
        [0., 10., 100.],
        [0., 10., 100.]], dtype=torch.float32)
    print(f'logits: {logits}')

    repeat_logits = torch.repeat_interleave(logits.unsqueeze(dim=-1), 20, dim=-1)
    repeat_logits[0] += torch.rand([3, 20])
    repeat_logits[1] += 10. * torch.rand([3, 20])
    repeat_logits[2] += torch.rand([3, 20])
    repeat_logits[3] += 10. * torch.rand([3, 20])

    scores = scorer.calc_scores_from_repeat_logits(repeat_logits)
    print(f'scores: {scores}')
