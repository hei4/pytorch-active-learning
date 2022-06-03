import torch

class BaseSampler:
    def __init__(self, scorer, num_samples) -> None:
        self.scorer = scorer
        self.num_samples = num_samples
        self.scores = []
    
    def scoring(self, features):
        scores_subset = self.scorer(features)
        self.scores.append(scores_subset)

    def sampling(self):
        self.scores = torch.cat(self.scores)
        indices = torch.argsort(self.scores, descending=True)
        sampling_indices = indices[:self.num_samples]
        rest_indices = indices[self.num_samples:]
        return sampling_indices, rest_indices

    def reset(self):
        self.scores = []
        self.indices = []


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parents[1]))

    from scorers.least_scorer import LeastConfidenceScorer

    scorer = LeastConfidenceScorer()
    sampler = UncertaintySampler(scorer, num_samples=2)

    logits = torch.tensor([
        [-1., 0., 1.],
        [0., 1., 2.],
        [0., 1., 10.],
        [0., 10., 100.]], dtype=torch.float32)
    print(f'logits: {logits}')

    sampler.scoring(logits)

    sampling_indices, rest_indices = sampler.sampling()
    print(f'sampling: [{sampling_indices}]')
    print(f'rest:     [{rest_indices}]')
