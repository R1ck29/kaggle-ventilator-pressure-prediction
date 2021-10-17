import torch
import torch.nn as nn

class L1Loss_masked(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, y, u_out):

        mask = 1 - u_out
        mae = torch.abs(mask * (y - preds))
        mae = torch.sum(mae) / torch.sum(mask)

        return mae