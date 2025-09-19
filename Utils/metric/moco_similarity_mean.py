import torch

def Positive_Negative_Mean(q, k, queue):
    """ Pos Neg Mean """
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
    
    return l_pos.mean().tolist(), l_neg.mean().tolist()

