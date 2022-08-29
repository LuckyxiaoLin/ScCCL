import torch.nn as nn


class DataAug(nn.Module):
    def __init__(self, dropout=0.9):
        super(DataAug, self).__init__()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        aug_data = self.drop(x)

        return aug_data