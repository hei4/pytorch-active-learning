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

        scores = []
        for i in range(logits.shape[1]):
            score = np.interp(
                # torch.mean(logits, dim=1).cpu().numpy(),
                # torch.min(logits, dim=1).values.cpu().numpy(),
                logits[:, i].cpu().numpy(),
                self.references[:, i].cpu().numpy(),
                self.rank)
            scores.append(score)
        scores = np.mean(np.stack(scores, axis=1), axis=1)

        return {'score': torch.tensor(scores)}
    
    def regist_features(self, features:torch.Tensor):
        self.net.eval()
        with torch.inference_mode():
            logits = self.net(features)
            # self.references.append(torch.mean(logits, dim=1))
            # self.references.append(torch.min(logits, dim=1).values)
            self.references.append(logits)
    
    def post_process(self):
        # self.references = torch.cat(self.references).sort().values
        self.references = torch.cat(self.references)
        for i in range(self.references.shape[1]):
            self.references[:, i] = self.references[:, i].sort().values

        self.rank = np.linspace(1., 0., len(self.references))
    
    def reset(self):
        self.references = []
        self.rank = np.empty(0)
