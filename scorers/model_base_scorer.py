import torch

class ModelBaseScorer:
    def __init__(self, net) -> None:
        self.net = net

    def __call__(self, features:torch.Tensor):
        self.net.eval()
        with torch.inference_mode():
            logits = self.net(features)
            scores = self.calc_scores_from_logits(logits)
        return {'score': scores}