import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative):
        pos_dist = self.calc_euclidean(anchor, positive)
        neg_dist = self.calc_euclidean(anchor, negative)

        losses = torch.relu(pos_dist - neg_dist + self.margin)
        #loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return losses.mean()