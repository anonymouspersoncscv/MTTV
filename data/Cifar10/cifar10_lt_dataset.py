import pytorch_lightning as pl
import torchvision
from data.MTTVModel_lt_Data import MTTVDataFromCIFAR10LT
from data.Simclr_lt_Data import SimclrDataFromCIFAR10LT
from data.Moco_lt_Data import MocoDataFromCIFAR10LT

from torch.utils.data import DataLoader
from Preprocess.Preprocess import Preprocess
import numpy as np

from Preprocess.MTTVDataAugmentation import MTTVDataAugmentation
from Preprocess.MocoV2DataAugmentation import MocoV2DataAugmentation
from Preprocess.SimclrDataAugmentation import SimclrDataAugmentation

def Get_Augmentation(model_name):
    
    model_function = model_name + "DataAugmentation"
    exec(f"generated_model = {model_function}", globals())
    return globals()['generated_model']

class Cifar10_DataModuleLT(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        
        self.data_dir = config.dataset.data_dir
        self.crop_max = config.dataset.crop_max
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.dataset = config.dataset.name
        self.image_size = config.dataset.image_size
        self.drop_last = config.dataset.drop_last
        self.mean = config.dataset.mean
        self.std = config.dataset.std
        self.imb_factor = config.dataset.imbalance_factor
        self.imb_type = config.dataset.imbalance_type
        self.model_name = config.model.name
            
        self.preprocess = Preprocess(image_size = self.image_size)
        if(self.model_name != "MocoV2"):
            self.K = config.model.K
        
        filter = int(config.model.gaussian_factor * self.image_size)
        if(filter % 2 == 0):
            kernel_size = filter + 1
        else:
            kernel_size = filter

        augmentation = Get_Augmentation(self.model_name)
        self.transform = augmentation(image_size = self.image_size,
                                           kernel_size = kernel_size,
                                           crop_max = self.crop_max,
                                           mean = self.mean,
                                           std = self.std)

    def setup(self, stage):
        
        if(self.model_name == "MTTV"):
            print("MTTV Data")
            self.train_set = MTTVDataFromCIFAR10LT(K=self.K,
                                     root = self.data_dir,
                                     transform = torchvision.transforms.ToTensor(),
                                     imb_factor = self.imb_factor,
                                     imb_type = self.imb_type,
                                     train = True,
                                     download = False)
            class_list = self.train_set.get_cls_num_list()
            print(class_list)
        
        if(self.model_name == "Simclr"):
            print("Simclr Data")
            self.train_set = SimclrDataFromCIFAR10LT(K=self.K,
                                     root = self.data_dir,
                                     transform = torchvision.transforms.ToTensor(),
                                     imb_factor = self.imb_factor,
                                     imb_type = self.imb_type,
                                     train = True,
                                     download = False)
            class_list = self.train_set.get_cls_num_list()
            print(class_list)

        if(self.model_name == "MocoV2"):
            print("Moco Data")
            self.train_set = MocoDataFromCIFAR10LT(root = self.data_dir,
                                     transform = torchvision.transforms.ToTensor(),
                                     imb_factor = self.imb_factor,
                                     imb_type = self.imb_type,
                                     train = True,
                                     download = False,
                                     )
            class_list = self.train_set.get_cls_num_list()
            print(class_list)
        
    def on_after_batch_transfer(self, batch, dataloader_idx):
    
        if(self.model_name == "MTTV"):     
            data, data_transform = batch
            d = data.size()
            train_x = data.view(d[0]*d[1], d[2],d[3], d[4])
            train_x_transform = data_transform.view(d[0]*d[1], d[2],d[3], d[4])
            train_x = self.transform.normalize(train_x)
            train_x_transform = self.transform.transforms(train_x_transform)
            return train_x,train_x_transform
        
        if(self.model_name == "Simclr"):
            data, target = batch        
            d = data.size()
            train_x = data.view(d[0]*self.K, d[2],d[3], d[4])
            train_x = self.transform.transforms(train_x)
            
            return train_x,target
        
        if(self.model_name == "Moco" or self.model_name == "MocoV2"):
            data_1, data_2,_ = batch        
            train_x = self.transform.transforms(data_1)
            train_y = self.transform.transforms(data_2)
            
            return train_x,train_y
        
    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = True,
                          pin_memory = False,
                          drop_last=self.drop_last,
                          )
        
        
    '''
    
    def val_dataloader(self):
        #return self.dataloaders['val']
        
        return DataLoader(self.val_set,
                      batch_size = self.batch_size,
                      num_workers = self.num_workers,
                      shuffle = False,
                      pin_memory = True,
                      drop_last=False)
    '''    
