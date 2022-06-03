import torch

class MarginConfidenceScorer:
    def __call__(self, logits:torch.Tensor):
        probabilities = torch.softmax(logits, dim=1)
        topk_prob = torch.topk(probabilities, k=2, dim=1).values
        diff_prob = topk_prob[:, 0] - topk_prob[:, 1]

        return 1. - diff_prob


if __name__ == '__main__':
    scorer = MarginConfidenceScorer()

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
