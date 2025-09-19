#import config
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from torch import nn
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import torch.distributed as dist
import utils

def prepare_data_features(model, dataset, config, mode="Train"):
    
    #if(config.dataset.name != "Imagenet"):
    device = config.training.gpu
    if(config.transfer_learning.transfer_learning):
        network = deepcopy(model)
    else:
        network = deepcopy(model.net)
    
    network.eval()
    network.to(device)

    dataloader = DataLoader(dataset, batch_size=config.training.batch_size,
                            num_workers = config.training.num_workers, 
                            shuffle = False,
                            drop_last = False)
    fetaures = []
    labels = []
    for images, targets in tqdm(dataloader):
        images = images.to(device)
        images_features = network(images)
        fetaures.append(images_features.detach().cpu())
        labels.append(targets)
    
    features = torch.cat(fetaures, dim=0)
    labels = torch.cat(labels, dim=0)

    labels, idx = labels.sort()
    features = features[idx]

    return TensorDataset(features, labels)
    
    '''
    Multi-GPU code for Feature Extraction (Useful for Imagenet) 
    '''
    '''
    else:
        if(mode == "test"):
            checkpoint_dir = utils.GetCheckpointDir(config, train_mode="pretrain")
            save_path = config.dataset.save_path + "/" + checkpoint_dir + "/features_"+mode+".pt"
            
            if os.path.exists(save_path):
                print("test Data loading...")
                cache = torch.load(save_path)
                features, labels = cache['features'], cache['labels']
            else:
                device = config.training.gpu
                if(config.transfer_learning.transfer_learning):
                    network = deepcopy(model)
                else:
                    network = deepcopy(model.net)
                #network.fc = nn.Identity()
                network.eval()
                network.to(device)

                dataloader = DataLoader(dataset, batch_size=config.training.batch_size,
                                        num_workers = config.training.num_workers, 
                                        shuffle = False,
                                        drop_last = False)
                features = []
                labels = []
                for images, targets in tqdm(dataloader):
                    images = images.to(device)
                    images_features = network(images)
                    features.append(images_features.detach().cpu())
                    labels.append(targets)
                
                features = torch.cat(features, dim=0)
                labels = torch.cat(labels, dim=0)

                labels, idx = labels.sort()
                features = features[idx]

                # Save only on main process
                torch.save({'features': features, 'labels': labels}, save_path)

            if(features == [] or labels == []):
                print("Either features or labels are empty!")
            #dist.destroy_process_group()
            return TensorDataset(features, labels)
        
        else:
            checkpoint_dir = utils.GetCheckpointDir(config, train_mode="pretrain")
            save_path = config.dataset.save_path + checkpoint_dir + "/features.pt"
            trainer = model.trainer
            device = torch.device(f"cuda:{torch.distributed.get_rank()}")
            is_rank_zero = trainer.is_global_zero
            local_rank = torch.distributed.get_rank()
            world_size = trainer.world_size

            # Check if features are cached (only on rank 0)
            if is_rank_zero and os.path.exists(save_path):
                cache = torch.load(save_path)
                features, labels = cache['features'], cache['labels']
                print("Features Loaded...")
            else:
                print(save_path)
                print("Features does not exist ...")
                # Set model to evaluation mode and remove final classification layer
                feature_extractor = deepcopy(model.net)
                feature_extractor.fc = nn.Identity()
                feature_extractor.eval()
                feature_extractor.to(device)
                # Wrap the model with DistributedDataParallel
                feature_extractor = DDP(feature_extractor, device_ids=[local_rank], output_device=local_rank)
                    
                # Prepare DataLoader with DistributedSampler
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
                dataloader = DataLoader(dataset, batch_size=config.training.batch_size,
                                        num_workers=config.training.num_workers,
                                        sampler=sampler,shuffle=False, drop_last=False)

                features_list, labels_list = [], []
                for images, targets in tqdm(dataloader, desc=f"[Rank {trainer.global_rank}] Extracting"):
                    images = images.to(device)
                    with torch.no_grad():
                        feats = feature_extractor(images)
                    features_list.append(feats.cpu())
                    labels_list.append(targets)

                local_features = torch.cat(features_list, dim=0).to(local_rank).contiguous()
                local_labels = torch.cat(labels_list, dim=0).to(local_rank).contiguous()

                # Allocate empty tensors for all_gather
                feature_list = [torch.zeros_like(local_features) for _ in range(world_size)]
                label_list = [torch.zeros_like(local_labels) for _ in range(world_size)]

                # Perform all_gather
                dist.all_gather(feature_list, local_features)
                dist.all_gather(label_list, local_labels)

                # Remove duplicates if gathered in a non-distributed-aware way
                features = []
                labels = []
                if is_rank_zero:
                    features = torch.cat(features_list, dim=0)
                    labels = torch.cat(labels_list, dim=0)
                    labels, idx = labels.sort()
                    features = features[idx]

                    # Save only on main process
                    torch.save({'features': features, 'labels': labels}, save_path)

            # Ensure all ranks wait until file is written
            if trainer.strategy.world_size > 1:
                torch.distributed.barrier()
            
            dist.destroy_process_group()
            if(features == [] or labels == []):
                print("Either features or labels are empty!")
            features = torch.Tensor(features)
            labels = torch.Tensor(labels)
            return TensorDataset(features, labels)
    '''
