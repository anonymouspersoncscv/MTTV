import pytorch_lightning as pl
from torchvision.datasets.fakedata import transforms
import config
from Feature_Extraction.features import prepare_data_features
from torch.utils.data import DataLoader
from data.MTTVModel_Data import Imagenet_Data
import torchvision

from Preprocess.Augmentations import model_transforms
from Preprocess.Preprocess import Preprocess
from Preprocess.DataAugmentation import DataAugmentation
from collections import Counter


class Imagenet_DataModule_le(pl.LightningDataModule):
    def __init__(self, model, config):
        super().__init__()

        self.config = config
        self.data_dir = config.dataset.data_dir
        #self.train_le_txt = config.dataset.train_le_txt_dir
        #self.val_le_txt = config.dataset.val_le_txt_dir
        
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.dataset = config.dataset.name
        self.image_size = config.dataset.image_size

        self.model = model
        #if(config.dataset.offset != 100):
        #    self.train_le_txt = None
        #    self.val_le_txt = None

        self.model_name = config.model.name
        if(self.model_name != "MocoV2"):
            self.K = config.model.K

    def prepare_data(self):     #Already Downloaded
        pass

    def setup(self, stage):
       
        transform = model_transforms(self.dataset, self.image_size)
        normalized_transform, _ = transform.GetTransform()
        train_set = Imagenet_Data(path=self.data_dir,
                                        transform = normalized_transform,
                                        split='train')


        test_set = Imagenet_Data(path = self.data_dir,
                                    transform = normalized_transform,
                                    split='val')
        
        if(stage == 'fit'):
            self.train_data = prepare_data_features(self.model, train_set, self.config,"train")
            features,labels = self.train_data.tensors
            
            print(features.shape)
            #print(labels.shape)
            print("max_labels:" + str(max(labels)+1))
            #self.test_data = prepare_data_features(self.model, test_set, self.config,"test")
            print("Linear Evaluation Data Loaded ...")
        #self.val_data = prepare_data_features(self.model, val_set)
        if(stage == 'test'):
            self.test_data = prepare_data_features(self.model, test_set, self.config,"test")
            features,labels = self.test_data.tensors
            
            print(features.shape)
            #print(labels.shape)
            print("max_labels:" + str(max(labels)+1))
            print("Testing Data Loaded ...")
        
     
    
    def train_dataloader(self):
        return DataLoader(self.train_data,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=True,
                           pin_memory=True)
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
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=False,
                           pin_memory=True)
