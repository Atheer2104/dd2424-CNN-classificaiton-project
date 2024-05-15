import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricCrossEntropyLearning(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0):
        super(SymmetricCrossEntropyLearning, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, y):
        # Cross Entropy (CE)
        ce_loss = F.cross_entropy(pred, y)

        # Reverse Cross Entropy (RCE)
        softmax_pred = F.softmax(pred, dim=1)
        softmax_pred = torch.clamp(softmax_pred, 1e-7, 1.0)
        one_hot_y = F.one_hot(y, pred.size(1)).float().to(self.device)
        one_hot_y = torch.clamp(one_hot_y, 1e-4, 1.0)
        rce_loss = torch.mean(-torch.sum(one_hot_y * torch.log(softmax_pred), dim=1))

        # Symetric Cross Entropy (SCE) = CE + RCE
        sce_loss = self.alpha * ce_loss + self.beta * rce_loss

        return sce_loss