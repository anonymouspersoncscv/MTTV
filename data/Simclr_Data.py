import torchvision
from PIL import Image
import os
import torch
from collections import Counter

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
Function to open an image and convert it into RGB
"""
def default_loader(path):
    return Image.open(path).convert('RGB')


class SimclrDataFromCIFAR10(torchvision.datasets.CIFAR10):
  def __init__(self, K, **kwds):
    super().__init__(**kwds)
    self.K = K # tot number of augmentations

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    pic = Image.fromarray(img)
    img_list = list()
    if self.transform is not None:
      for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)
    else:
        img_list = img
    data = stack(img_list,0)
    return data, target

class SimclrDataFromCIFAR100(torchvision.datasets.CIFAR100):
  def __init__(self, K, **kwds):
    super().__init__(**kwds)
    self.K = K # tot number of augmentations

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    pic = Image.fromarray(img)
    img_list = list()
    if self.transform is not None:
      for _ in range(self.K):
        img_transformed = self.transform(pic.copy())
        img_list.append(img_transformed)
    else:
        img_list = img
    data = stack(img_list,0)
    return data, target

class SimclrDataFromImagenet(torchvision.datasets.ImageNet):
    def __init__(self, K, path, split, **kwds):
        super().__init__(**kwds)
        self.K = K  # tot number of augmentations
        self.path = path
        self.split = split
        self.data = torchvision.datasets.ImageNet(root = self.path, split=self.split)

    def __getitem__(self, index):
        pic, target = self.data[index]
        img_list = list()
        for _ in range(self.K):
            img_transformed = self.transform(pic.copy())
            img_list.append(img_transformed)
        #else:
        #    raise Exception("transforms are missing...")

        data = stack(img_list,0) 
        return data, target
 
    def get_cls_num_list(self, indices):
        #import pdb
        #pdb.set_trace()
        labels = [self.data[i][1] for i in indices.tolist()]
        cls_num_list = Counter(labels)
        return cls_num_list

class Imagenet_Data(torchvision.datasets.ImageNet):
    def __init__(self, path, **kwds):
        super().__init__(**kwds)
        self.path = path
        self.data = torchvision.datasets.ImageNet(root = self.path)

    def __getitem__(self, index):
        pic, target = self.data[index]
        if self.transform is not None:
            pic = self.transform(pic)
        else:
            raise Exception("transforms are missing...")

        return pic, target

