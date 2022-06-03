from torch import nn

class MLP(nn.Module):
    def __init__(self, num_features=2, num_unit=64, num_classes=2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_unit),
            nn.ReLU(inplace=True),
            nn.Linear(num_unit, num_unit),
            nn.ReLU(inplace=True),
            nn.Linear(num_unit, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)
