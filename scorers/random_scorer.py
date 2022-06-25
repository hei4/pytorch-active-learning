import torch

class RandomScorer:
    def __call__(self, features:torch.Tensor):
        device = features.device
        return {'score': torch.full([len(features)], 0.5, device=device)}
    
    def reset(self):
        pass    # ダミー