import torch
import numpy as np

class ReferenceScorer:
    def __init__(self):
        self.references = []
        self.rank = np.empty(0)

    def __call__(self, logits:torch.Tensor):
        score = np.interp(
            torch.mean(logits, dim=1).cpu().numpy(), 
            self.references.cpu().numpy(),
            self.rank)
        return torch.tensor(score)
    
    def make_reference(self, logits:torch.Tensor):
        self.references.append(torch.mean(logits, dim=1))
    
    def make_rank(self):
        self.references = torch.cat(self.references).sort().values
        self.rank = np.linspace(1., 0., len(self.references))
    
    def reset(self):
        self.references = []
        self.rank = np.empty(0)
