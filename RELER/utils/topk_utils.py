import numpy as np
import torch

EPSILON = np.finfo(np.float32).tiny


# Source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html
class TopKOperator(torch.nn.Module):
    def __init__(self, hard=False):
        super(TopKOperator, self).__init__()
        self.hard = hard

    def forward(self, scores, k, tau=1.0, onehot_mask=None):
        device = scores.device
        m = torch.distributions.gumbel.Gumbel(
            torch.zeros_like(scores), torch.ones_like(scores)
        )
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        if onehot_mask is None:
            onehot_approx = torch.zeros_like(scores)
        else:
            onehot_approx = onehot_mask
        for i in range(k):
            khot_mask = torch.max(
                1.0 - onehot_approx, torch.tensor([EPSILON]).to(device)
            )
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res
