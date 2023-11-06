import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Linear, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, n_features//2),
            nn.ReLU(inplace=True),
            nn.Linear(n_features//2, n_classes)
        )

    def forward(self, x):
        return self.model(x)
