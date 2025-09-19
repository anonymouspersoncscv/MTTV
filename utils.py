import torchvision
import torch
import torch.nn as nn
import math
import os
from data.MTTVModel_Data import Imagenet_Data

def alignment_loss(embeddings, positive_pairs):
    """Minimizes distance between positive pairs."""
    z_i, z_j = embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]
    return (z_i - z_j).pow(2).sum(dim=1).mean()

def uniformity_loss(embeddings, t=2.0):
    """Encourages embeddings to be uniformly distributed."""
    sq_pdist = torch.pdist(embeddings, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_pdist).mean())

def GetCheckpointDir(config, train_mode):

    checkpoint_path = ""
    
    if(config.dataset.imbalance):
        checkpoint_path = os.path.join("results", "imbalance_" + train_mode, config.dataset.imbalance_type, config.model.name + config.model.mode, config.dataset.name, config.backbone.name)
    else:
        checkpoint_path = os.path.join("results", "balance_" + train_mode, config.model.name + config.model.mode, config.dataset.name, config.backbone.name)

    return checkpoint_path

def GetTensorboardDir(config, train_mode):

    result_folder = ""
    if(config.dataset.imbalance):
        result_folder = os.path.join("results", train_mode + "_logs", "imbalance", config.dataset.imbalance_type, config.model.name + config.model.mode, config.dataset.name)
    else:
        result_folder = os.path.join("results", train_mode + "_logs", "balanced", config.model.name + config.model.mode, config.dataset.name)

    return result_folder

def adjust_learning_rate(optimizer, init_lr, epoch, max_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / max_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

# warmup + cosine decay
def warmup_cosine_decay_leraning_rate(init_lr, current_epoch, warmup_epochs, warmup_lr, max_epochs):
    if current_epoch < warmup_epochs:
        # Linear warmup: Increase from warmup_lr to base_lr
        return warmup_lr + (init_lr - warmup_lr) * (current_epoch / warmup_epochs) / init_lr
    else:
        # Cosine decay
        decay_epochs = max_epochs - warmup_epochs
        decay_epoch = current_epoch - warmup_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_epoch / decay_epochs))
        return cosine_decay

def GetBackbone(backbone_name, dataset_name, prune=False, num_class=10):
    match backbone_name:
        case "resnet18":
            if(dataset_name == "Cifar10" or dataset_name == "Cifar100"):
                net = torchvision.models.resnet18(weights = None)
                net.maxpool = nn.Identity()
                net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                net.fc = nn.Identity()
            else:
                net = torchvision.models.resnet18(weights = None)
                net.fc = nn.Identity()

            return net
        
        case "resnet50":
            if(dataset_name == "Cifar10" or dataset_name == "Cifar100"):
                net = torchvision.models.resnet18(weights = None)
                net.maxpool = nn.Identity()
                net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                net.fc = nn.Identity()
            else:
                net = torchvision.models.resnet50(weights = None)
                net.fc = nn.Identity()
            
            return net

def Get_Dataset(dataset_name, data_dir, test_transform, batch_size, data_list=None, val_list=None):
    match dataset_name:
        case "Cifar10":
            train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=test_transform, download=True)
            test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
        case "Cifar100":
            train_data = torchvision.datasets.CIFAR100(root=data_dir, train=True, transform=test_transform, download=True)
            test_data = torchvision.datasets.CIFAR100(root=data_dir, train=False, transform=test_transform, download=True)
            
        case "Imagenet":
            train_data = Imagenet_Data(path = data_dir,
                                            transform = test_transform,
                                            #transform = torchvision.transforms.Compose([resize,torchvision.transforms.ToTensor(),normalize]),
                                            #root=data_dir, 
                                            split='train')


            test_data = Imagenet_Data(path = data_dir,
                                        transform = test_transform,
                                        #transform = torchvision.transforms.Compose([resize,torchvision.transforms.ToTensor(),normalize]),
                                        #root=data_dir, 
                                        split='val')

    memory_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return memory_data_loader, test_data_loader
