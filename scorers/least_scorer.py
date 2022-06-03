import torch

class LeastConfidenceScorer:
    def __call__(self, logits:torch.Tensor):
        probability = torch.softmax(logits, dim=1)
        most_confidence = torch.max(probability, dim=1).values
        num_labels = probability.size(dim=1)

        numerator = num_labels * (1. - most_confidence)
        denominator = num_labels - 1

        return numerator / denominator


if __name__ == '__main__':
    scorer = LeastConfidenceScorer()

    logits = torch.tensor([
        [-1., 0., 1.],
        [0., 1., 2.],
        [0., 1., 10.],
        [0., 10., 100.]], dtype=torch.float32)
    print(f'logits: {logits}')
    
    scores = scorer(logits)
    print(f'scores: {scores}')
