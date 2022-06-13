import torch

class RandomScorer:
    def __call__(self, logits:torch.Tensor):
        device = logits.device
        return {'score': torch.full([len(logits)], 0.5, device=device)}
    
    def reset(self):
        pass    # ダミー