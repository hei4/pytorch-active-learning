import torch
import math

class EntropyConfidenceScorer:
    def __call__(self, logits:torch.Tensor):
        probabilities = torch.softmax(logits, dim=1)
        prob_logs = probabilities * torch.log2(probabilities)
        numerator = -torch.sum(prob_logs, dim=1)
        denominator = math.log2(probabilities.size(1))

        return numerator / denominator


if __name__ == '__main__':
    scorer = EntropyConfidenceScorer()

    logits = torch.tensor([
        [-1., 0., 1.],
        [0., 1., 2.],
        [0., 1., 10.],
        [0., 10., 100.]], dtype=torch.float32)
    print(f'logits: {logits}')
    
    probabilities = torch.softmax(logits, dim=1)
    print(f'probabilities: {probabilities}')

    scores = scorer(logits)
    print(f'scores: {scores}')
