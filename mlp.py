import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, contrastive_dim, mlp_dim, dim):
        super(MLP, self).__init__()
        self.contrastive_dim = contrastive_dim
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.contrastive_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.dim),
        )

    def forward(self, x):
        latent_out = self.encoder(x)
        return latent_out