import torchvision.datasets as datasets

import pytorch_lightning as pl
from Preprocess.Augmentations import model_transforms
#import config
from Feature_Extraction.features import prepare_data_features
from torch.utils.data import DataLoader

from Preprocess.Preprocess import Preprocess
from Preprocess.DataAugmentation import DataAugmentation
from collections import Counter
import torchvision

"""
CIFAR-100 Dataset class - returns the pair of one image 
                          with its another random K counterpart
"""
class DataFromCIFAR100(torchvision.datasets.CIFAR100):
  def __init__(self, data_dir, **kwds):
    super().__init__(**kwds)
    self.data_dir = data_dir
    with open("labels_Cifar100.txt","r") as f:
        self.indices = [int(line.strip()) for line in f if line.strip()]
    self.full_dataset = torchvision.datasets.CIFAR100(root=self.data_dir,train=True)

  def __len__(self) -> int:
     return len(self.indices)

  def __getitem__(self,index):
    real_idx = self.indices[index]
    img,target = self.full_dataset[real_idx]
    if self.transform is not None:
       img = self.transform(img)

    return img, target


class Cifar100_DataModule_le(pl.LightningDataModule):
    def __init__(self, model=None, config=None):
        super().__init__()

        self.model = model
        self.model_name = config.model.name
        #self.K = config.dataset.K
        self.config = config
        self.data_dir = config.dataset.data_dir
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.dataset = config.dataset.name
        self.image_size = config.dataset.image_size

    def prepare_data(self):     #Already Downloaded
        pass

    def setup(self, stage):
        transform = model_transforms(self.dataset, self.image_size)
        normalized_transform, augmentation_transform = transform.GetTransform()
        
        train_set = datasets.CIFAR100(root = self.data_dir,
                                          transform = normalized_transform,
                                          train = True,
                                          download = False)
        
        #train_set, val_set = random_split(entire_set, [40000, 10000])

        test_set = datasets.CIFAR100(root = self.data_dir,
                                         transform = normalized_transform,
                                         train = False,
                                         download = False)
        
        if(stage == "fit"):
            self.train_data = prepare_data_features(self.model, train_set, self.config)
        #self.val_data = prepare_data_features(self.model, val_set)
        elif(stage == "test"):
            self.test_data = prepare_data_features(self.model, test_set, self.config)

    
    def train_dataloader(self):
        return DataLoader(self.train_data,
                      batch_size = self.batch_size,
                      num_workers = self.num_workers,
                      shuffle = True,
                      pin_memory = True)
        
    '''
    def val_dataloader(self):
        return DataLoader(self.val_data,
                      batch_size = self.batch_size,
                      num_workers = self.num_workers,
                      shuffle = False,
                      pin_memory = True)
    ''' 
    def test_dataloader(self):
        return DataLoader(self.test_data,
                      batch_size = self.batch_size,
                      num_workers = self.num_workers,
                      shuffle = False,
                      pin_memory = True)
        
