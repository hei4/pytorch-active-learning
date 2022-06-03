import numpy as np
from scipy import interpolate
import torch
from samplers.base import BaseSampler

class ReferenceSampler(BaseSampler):
    def __init__(self, scorer, num_samples) -> None:
        super().__init__(scorer, num_samples)
        self.reference_scores = []
    
    def reference_scoring(self, features):
        scores_subset = self.scorer(features)
        self.reference_scores.append(scores_subset)

    def sampling(self):
        self.reference_scores = torch.cat(self.reference_scores)
        self.reference_scores = self.reference_scores.sort().values

        reference_ranking = np.linspace(1., 0., len(self.reference_scores))
                
        self.scores = torch.cat(self.scores)

        ranking = torch.tensor(np.interp(
            self.scores.cpu().numpy(), 
            self.reference_scores.cpu().numpy(),
            reference_ranking))

        indices = torch.argsort(ranking, descending=True)
        sampling_indices = indices[:self.num_samples]
        rest_indices = indices[self.num_samples:]
        
        return sampling_indices, rest_indices

    def reset(self):
        self.scores = []
        self.indices = []
        self.reference_scores = []


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parents[1]))

    from scorers.reference_scorer import ReferenceScorer

    scorer = ReferenceScorer()
    sampler = ReferenceSampler(scorer, num_samples=2)

    refernce_logits = torch.tensor([
        [-1., 0., 1.],
        [9., 10., 11.],
        [99., 100., 101.]
    ], dtype=torch.float32)
    print(f'refernce logits: {refernce_logits}')

    sampler.reference_scoring(refernce_logits)

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
