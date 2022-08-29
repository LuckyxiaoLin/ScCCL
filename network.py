import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):

    def __init__(self, encoder_q, encoder_k, instance_projector, cluster_projector, class_num,
                 m=0.2):
        super(Network, self).__init__()

        self.cluster_num = class_num
        self.m = m

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.instance_projector = instance_projector

        self.cluster_projector = nn.Sequential(
            cluster_projector,
            nn.Softmax(dim=1)
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, cell_q, cell_k):
        q = self.encoder_q(cell_q)
        q_instance = normalize(self.instance_projector(q), dim=1)
        q_cluster = self.cluster_projector(q)

        if cell_k is None:
            return q_instance, q_cluster, None, None

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(cell_k)
            k_instance = normalize(self.instance_projector(k), dim=1)
            k_cluster = self.cluster_projector(k)

        return q_instance, q_cluster, k_instance, k_cluster