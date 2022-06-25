import torch
import numpy as np

class ReferenceScorer:
    def __init__(self, net):
        self.net = net
        self.references = []
        self.rank = np.empty(0)

    def __call__(self, features:torch.Tensor):
        self.net.eval()
        with torch.inference_mode():
            logits = self.net(features)

        score = np.interp(
            torch.mean(logits, dim=1).cpu().numpy(), 
            self.references.cpu().numpy(),
            self.rank)

        return {'score': torch.tensor(score)}
    
    def regist_features(self, features:torch.Tensor):
        self.net.eval()
        with torch.inference_mode():
            logits = self.net(features)
            self.references.append(torch.mean(features, dim=1))
    
    def post_process(self):
        self.references = torch.cat(self.references).sort().values
        self.rank = np.linspace(1., 0., len(self.references))
    
    def reset(self):
        self.references = []
        self.rank = np.empty(0)
