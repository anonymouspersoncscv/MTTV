import torch
import torch.nn.functional as F

def infonce_loss(q, k, queue, temperature=0.07):
    """ InfoNCE loss """
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
    loss = F.cross_entropy(logits, labels)
    return loss
