import pytorch_lightning as pl
import torchvision
from data.MTTVModel_lt_Data import MTTVDataFromImagenetLT
from data.Simclr_lt_Data import SimclrDataFromImagenetLT
from data.Moco_lt_Data import MocoDataFromImagenetLT
from torch.utils.data import DataLoader, SubsetRandomSampler
from Preprocess.Preprocess import Preprocess
import numpy as np

from Preprocess.MTTVDataAugmentation import MTTVDataAugmentation
from Preprocess.SimclrDataAugmentation import SimclrDataAugmentation
from Preprocess.MocoV2DataAugmentation import MocoV2DataAugmentation
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

def Get_Augmentation(model_name):
    
    model_function = model_name + "DataAugmentation"
    exec(f"generated_model = {model_function}", globals())
    return globals()['generated_model']

class Imagenet_DataModuleLT(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
         
        self.data_dir = config.dataset.data_dir
        self.text_dir = config.dataset.text_dir
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.dataset = config.dataset.name
        self.image_size = config.dataset.image_size
        self.crop_max = config.dataset.crop_max
        self.mean = config.dataset.mean
        self.std = config.dataset.std
        self.drop_last = config.dataset.drop_last
        self.model_name = config.model.name
        self.dataset_size = config.dataset.dataset_size
        self.sample_size = config.dataset.sample_size
        self.device = config.training.gpu
        #self.classes = config.dataset.num_classes 
        self.preprocess = Preprocess(image_size = self.image_size)
         
        if(self.model_name != "MocoV2"):
            self.K = config.model.K
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
        resize = torchvision.transforms.Resize(size=(self.image_size, self.image_size))
        subset_idx = None
        if(self.sample_size < 1):
            labels = [] 
            with open(self.text_dir) as f:
                for line in f:
                    labels.append(int(line.split()[1]))
            
            targets = np.array(labels)
            sss = StratifiedShuffleSplit(n_splits=1, train_size=int(self.sample_size * self.dataset_size), random_state=42)
            indices = list(range(len(targets)))
            for subset_idx, _ in sss.split(indices, targets):
                self.subsetSampler = SubsetRandomSampler(subset_idx)
                subset_labels = targets[subset_idx]
                labels_count = Counter(subset_labels)

            labels_list = [j for _,j in labels_count.items()]
            print(labels_list)
            print("Sample size: " + str(sum(labels_list)))
             
        self.subset_idx = subset_idx
        
        if(self.model_name == "MTTV"):
            print("MTTV Data")
            self.train_set = MTTVDataFromImagenetLT(K=1,
                                     root = self.data_dir,
                                     txt = self.text_dir,
                                     transform = torchvision.transforms.Compose([resize, torchvision.transforms.ToTensor()]),
                                     subset_indices=subset_idx,
                                    )
            #import pdb
            #pdb.set_trace()
            if(self.sample_size == 1):
                class_list = self.train_set.get_cls_num_list()
                print(class_list)
        
        if(self.model_name == "Simclr" or self.model_name =="SimclrV2"):
            print("Simclr Data")
            self.train_set = SimclrDataFromImagenetLT(K=self.K, #path = self.data_dir,
                                     root = self.data_dir,
                                     txt = self.text_dir,
                                     transform = torchvision.transforms.Compose([resize, torchvision.transforms.ToTensor()]),
                                     subset_indices=subset_idx,
                                    )
            if(self.sample_size == 1):
                class_list = self.train_set.get_cls_num_list()
                print(class_list)
        
        if(self.model_name == "Moco" or self.model_name == "MocoV2"):
            print("Moco Data")
            self.train_set = MocoDataFromImagenetLT(root = self.data_dir, #path = self.data_dir,
                                     transform = torchvision.transforms.Compose([resize, torchvision.transforms.ToTensor()]),
                                     txt = self.text_dir,
                                     subset_indices=subset_idx,
                                     )
            if(self.sample_size == 1):
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
        
        if(self.model_name == "MocoV2"):
            data_1, data_2, _ = batch        
            train_x = self.transform.transforms(data_1)
            train_y = self.transform.transforms(data_2)
            
            return train_x,train_y
        
        
    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          drop_last=self.drop_last,
                          shuffle = True,
                          pin_memory = False,
                          #sampler=self.subsetSampler,
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
