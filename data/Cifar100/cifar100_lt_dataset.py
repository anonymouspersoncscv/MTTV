import pytorch_lightning as pl
import torchvision
import torch

from data.MTTVModel_lt_Data import MTTVDataFromCIFAR100LT
from data.Simclr_lt_Data import SimclrDataFromCIFAR100LT
from data.Moco_lt_Data import MocoDataFromCIFAR100LT

from torch.utils.data import DataLoader
from Preprocess.MTTVDataAugmentation import MTTVDataAugmentation
from Preprocess.SimclrDataAugmentation import SimclrDataAugmentation
from Preprocess.MocoV2DataAugmentation import MocoV2DataAugmentation

def Get_Augmentation(model_name):
    
    model_function = model_name + "DataAugmentation"
    exec(f"generated_model = {model_function}", globals())
    return globals()['generated_model']

class Cifar100_DataModuleLT(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
         
        self.model_name = config.model.name

        self.data_dir = config.dataset.data_dir
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.normalization = config.training.normalization
        self.drop_last = config.dataset.drop_last
        self.dataset = config.dataset.name
        self.image_size = config.dataset.image_size
        self.crop_max = config.dataset.crop_max
        self.mean = config.dataset.mean
        self.std = config.dataset.std
        self.imb_factor = config.dataset.imbalance_factor
        self.imb_type = config.dataset.imbalance_type

        if(self.model_name != "MocoV2"):
            self.K = config.model.K

        #self.preprocess = Preprocess(image_size = self.image_size)
        if(self.normalization):
            print("Normalization, Augmentation")
        else:
            print("Augmentation, Augmentation")

        filter = int(0.1 * self.image_size)
        if(filter % 2 == 0):
            kernel_size = filter - 1
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
            self.train_set = MTTVDataFromCIFAR100LT(K=self.K,
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
            self.train_set = SimclrDataFromCIFAR100LT(K=self.K,
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
            self.train_set = MocoDataFromCIFAR100LT(root = self.data_dir,
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
            if(self.normalization):
                train_x = self.transform.normalize(train_x)
            else:
                train_x = self.transform.transforms(train_x)
            train_x_transform = self.transform.transforms(train_x_transform)
            return train_x,train_x_transform
                    
        if(self.model_name == "Simclr"):
            data, target = batch        
            d = data.size()
            '''Used in Simclr + NA'''
            if(self.normalization):
                x1 = data[:, 0]
                x2 = data[:, 1]
                x1 = self.transform.normalize(x1)
                x2 = self.transform.transforms(x2)
                data = torch.stack((x1,x2),dim=1)
                train_x = data.view(d[0]*d[1], d[2],d[3], d[4])
            else:
                train_x = data.view(d[0]*d[1], d[2],d[3], d[4])
                train_x = self.transform.transforms(train_x)
            return train_x,target
        
        if(self.model_name == "MocoV2"):
            data_1, data_2, _ = batch        
            train_x = self.transform.transforms(data_1)
            train_y = self.transform.transforms(data_2)
            
            return train_x,train_y
        
    def train_dataloader(self):
        
        return DataLoader(self.train_set,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = True,
                          drop_last=self.drop_last,
                          pin_memory = False,
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
