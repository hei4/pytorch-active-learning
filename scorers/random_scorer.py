import torch

class RandomConfidenceScorer:
    def __call__(self, logits:torch.Tensor):
        device = logits.device
        return torch.full([len(logits)], 0.5, device=device)