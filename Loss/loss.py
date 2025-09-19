import torch
import torch.nn.functional as F

def xent_loss(x, t=0.5, eps=1e-8):
    # Estimate cosine similarity
    n = torch.norm(x, p=2, dim=1, keepdim=True)
    x = (x @ x.t()) / (n * n.t()).clamp(min=eps)
    x = torch.exp(x /t)

    # Put positive pairs on the diagonal
    idx = torch.arange(x.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]

    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))

    log_value = -torch.log(x.mean())

    return log_value

def Our_xent_loss(x, lower_bound, upper_bound, t=0.5, eps=1e-8):
    # Estimate cosine similarity
    n = torch.norm(x, p=2, dim=1, keepdim=True)
    x = (x @ x.t()) / (n * n.t()).clamp(min=eps)

    '''Mask to eliminate extreme features'''

    '''Prefer this for Cifar10 and Cifar100'''
    mask = ((x > upper_bound) | (x < -upper_bound)) | ((x < lower_bound) & (x > -lower_bound))

    '''mask = ((x < lower_bound) & (x > -lower_bound))     <-- Prefer this for Imagenet and their subsets'''
    x = torch.exp(x /t)
    
    # Put positive pairs on the diagonal
    idx = torch.arange(x.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]

    # Put positive pairs on the diagonal
    idx1 = torch.arange(x.size()[0])
    idx1[::2] += 1
    idx1[1::2] -= 1
    mask = mask[idx1]
    
    '''Elimination'''
    x[mask] = 0
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))

    log_value = -torch.log(x.mean())

    return mask.sum().item(), log_value

def nt_xent_loss(z1, z2, temperature=0.5):
    """ NT-Xent loss """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2*N, device=device, dtype=torch.int64)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)
