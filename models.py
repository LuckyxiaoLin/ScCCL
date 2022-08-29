import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    def __init__(self, dims):
        super(BaseEncoder, self).__init__()
        self.dims = dims
        self.n_stacks = len(self.dims)  # -1
        enc = [nn.Linear(self.dims[0], self.dims[1]), nn.BatchNorm1d(self.dims[1]), nn.ReLU(),
               nn.Linear(self.dims[1], self.dims[2]), nn.BatchNorm1d(self.dims[2]), nn.ReLU(),
               nn.Linear(self.dims[2], self.dims[3]), nn.BatchNorm1d(self.dims[3]), nn.ReLU()]

        self.encoder = nn.Sequential(*enc)
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        latent_out = self.encoder(x)
        latent_out = F.normalize(latent_out, dim=1)

        return latent_out