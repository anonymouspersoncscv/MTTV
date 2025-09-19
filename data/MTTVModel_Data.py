from posix import read
import torch
import torchvision
from PIL import Image
import os
import random
import numpy as np
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset

"""
Function to open an image and convert it into RGB
"""
def default_loader(path):
    return Image.open(path).convert('RGB')



def stack(data, dim=0):
  shape = data[0].shape  # need to handle empty list
  shape = shape[:dim] + (len(data),) + shape[dim:]
  x = torch.cat(data, dim=dim)
  x = x.reshape(shape)
  # need to handle case where dim=-1
  # which is not handled here yet
  # but can be done with transposition
  return x
"""
CIFAR-10 Dataset class - returns the pair of one image 
                          with its random K counterparts
"""
class MTTVDataFromCIFAR10(torchvision.datasets.CIFAR10):
  def __init__(self, K,**kwds):
    super().__init__(**kwds)
    self.K = K              # tot number of augmentations

  def __getitem__(self,index):
    img,_=self.data[index], self.targets[index]
    pic = Image.fromarray(img)
    img_list = []
    img_trans_list = []
    #if self.transform1 is not None and self.transform2 is not None:
    for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)

        randNumber = random.randint(0, len(self.data)-1)
        img2,_ = self.data[randNumber], self.targets[randNumber]
        pic2 = Image.fromarray(img2)
        
        img_transformed = self.transform(pic2.copy())
        img_list.append(img_transformed)
        
        img_transformed = self.transform(pic2.copy())
        img_trans_list.append(img_transformed)

        img_transformed = self.transform(pic.copy())
        img_trans_list.append(img_transformed)
        
    #else:
      #raise Exception("transforms are missing...")
    
    data = stack(img_list,0)
    data_transform = stack(img_trans_list,0)

    
    return data, data_transform


"""
CIFAR-100 Dataset class - returns the pair of one image 
                          with its random K counterparts
"""
class MTTVDataFromCIFAR100(torchvision.datasets.CIFAR100):
  def __init__(self, K,**kwds):
    super().__init__(**kwds)
    self.K = K              # tot number of augmentations

  def __getitem__(self,index):
    img,_=self.data[index], self.targets[index]
    pic = Image.fromarray(img)
    img_list = []
    img_trans_list = []
    #if self.transform1 is not None and self.transform2 is not None:
    for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)
        #img_list = torch.cat((img_list,img_transformed),1)

        randNumber = random.randint(0, len(self.data)-1)
        img2,_ = self.data[randNumber], self.targets[randNumber]
        pic2 = Image.fromarray(img2)
        
        img_transformed = self.transform(pic2.copy())
        img_list.append(img_transformed)
        #img_list = torch.cat((img_list,img_transformed),1)
        
        img_transformed = self.transform(pic2.copy())
        img_trans_list.append(img_transformed)
        #img_trans_list = torch.cat((img_trans_list,img_transformed),1)

        img_transformed = self.transform(pic.copy())
        img_trans_list.append(img_transformed)
        #img_trans_list = torch.cat((img_trans_list,img_transformed),1)
        
    #else:
      #raise Exception("transforms are missing...")
    
    data = stack(img_list,0)
    data_transform = stack(img_trans_list,0)

    
    return data, data_transform, #index, randNumber

"""
Imagenet-LT Dataset class - returns the pair of one image 
                          with its random K counterparts
"""
class MTTVDataFromImagenet(torchvision.datasets.ImageNet):
    def __init__(self, K, path, split, **kwds):
        super().__init__(**kwds)
        self.K = K  # tot number of augmentations
        self.path = path
        self.split = split
        self.data = torchvision.datasets.ImageNet(root = self.path, split=self.split)

    def __getitem__(self, index):
        pic, target = self.data[index]
        img_list = []
        img_trans_list = []
        #if self.transform is not None:
        for _ in range(self.K):
            img_transformed = self.transform(pic.copy())
            img_list.append(img_transformed)

            randNumber = random.randint(0, len(self.data)-1)
            pic2, target2 = self.data[randNumber]

            img_transformed = self.transform(pic2.copy())
            img_list.append(img_transformed)

            img_transformed = self.transform(pic2.copy())
            img_trans_list.append(img_transformed)

            img_transformed = self.transform(pic.copy())
            img_trans_list.append(img_transformed)
        #else:
        #    raise Exception("transforms are missing...")

        data = stack(img_list,0)
        data_transform = stack(img_trans_list,0)

        
        return data, data_transform

"""
Imagenet Dataset class - returns the pair of one image,label
"""
class Imagenet_Data(Dataset):
    def __init__(self, path, split='train', transform=None):
        """
        path: root path to ImageNet directory
        split: 'train' or 'val'
        transform: torchvision transforms to apply
        """
        self.data = ImageNet(root=path, split=split, transform=transform)
        self.classes = self.data.classes
        self.class_to_idx = self.data.class_to_idx
        self.targets = [label for _,label in self.data.samples]
        print(f"Loaded {split} dataset with {len(self.data)} samples.")

    def __getitem__(self, index):
        return self.data[index]  # already returns (image, target)

    def __len__(self):
        return len(self.data)

