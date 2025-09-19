import torch
import torchvision
from PIL import Image
import os
import random
import numpy as np
import json
import utils
from scipy.stats import pareto
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
CIFAR10-LT Dataset class - returns the pair of one image 
                          with its random K counterparts
"""


class MTTVDataFromCIFAR10LT(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, K, imb_type='exp', imb_factor=0.01, rand_number=0, **kwds):
        super().__init__(**kwds)
        #super(MTTVDataFromCIFAR10LT, self).__init__(root, train, download)
        self.K = K
        #self.transform1 = transform1
        #self.transform2 = :qtransform2transform2
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            #import pdb
            #pdb.set_trace()
            img = self.data[selec_idx, ...]
            new_data.append(img)
            new_targets.extend([the_class, ] * the_img_num)
        
        new_data = np.vstack(new_data)
        
        self.data = new_data
        self.targets = new_targets
        
    #For MTTVs
    
    def __getitem__(self,index):
        img,_=self.data[index], self.targets[index]
        pic = Image.fromarray(img)
        img_list = []
        img_trans_list = []
        
        if self.transform is not None:
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
        else:
          raise Exception("transforms are missing...")
        
        data = stack(img_list, 0)
        data_trans = stack(img_trans_list, 0)

        return data, data_trans

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


"""
CIFAR100-LT Dataset class - returns the pair of one image 
                          with its random K counterparts
"""

class MTTVDataFromCIFAR100LT(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, K, imb_type='exp', imb_factor=0.01, rand_number=0, **kwds):
        super().__init__(**kwds)
        #super(MTTVDataFromCIFAR10LT, self).__init__(root, train, download)
        self.K = K
        #self.transform1 = transform1
        #self.transform2 = transform2
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            #import pdb
            #pdb.set_trace()
            img = self.data[selec_idx, ...]
            new_data.append(img)
            new_targets.extend([the_class, ] * the_img_num)
        
        new_data = np.vstack(new_data)
        
        self.data = new_data
        self.targets = new_targets
    
    #For MTTVs
    
    def __getitem__(self,index):
        img,target=self.data[index], self.targets[index]
        pic = Image.fromarray(img)
        img_list = []
        img_trans_list = []
        
        if self.transform is not None:
          for _ in range(self.K):
            img_transformed = self.transform(pic.copy())
            img_list.append(img_transformed)
            randNumber = random.randint(0, len(self.data)-1)
            img2,randTarget = self.data[randNumber], self.targets[randNumber]
            pic2 = Image.fromarray(img2)

            img_transformed = self.transform(pic2.copy())
            img_list.append(img_transformed)

            img_transformed = self.transform(pic2.copy())
            img_trans_list.append(img_transformed)

            img_transformed = self.transform(pic.copy())
            img_trans_list.append(img_transformed)
        else:
          raise Exception("transforms are missing...")
        
        data = stack(img_list, 0)
        data_trans = stack(img_trans_list, 0)

        return data, data_trans

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

"""
Imagenet-LT Dataset class - returns the pair of one image 
                          with its random K counterparts
"""

class MTTVDataFromImagenetLT(torchvision.datasets.ImageNet):

    def __init__(self, K, root, txt, transform=None, subset_indices=None):
        self.img_path = []
        self.labels = []
        self.K = K
        self.cls_num = 1000
        self.transform = transform
        self.subset_indices = subset_indices
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
    def __len__(self):
        if(self.subset_indices is None):
            return len(self.img_path)
        else:
            return len(self.subset_indices)

    def __getitem__(self, index):
        if(self.subset_indices is not None):
            real_idx = self.subset_indices[index]
        else:
            real_idx = index
        path, target = self.img_path[real_idx], self.labels[real_idx]
        pic = Image.open(path).convert("RGB")
        img_list = []
        img_trans_list = []
        for _ in range(self.K):
            img_transformed = self.transform(pic.copy())
            img_list.append(img_transformed)
            
            if(self.subset_indices is None):
                randNumber = np.random.choice(len(self.img_path))
            else:
                randNumber = np.random.choice(self.subset_indices)

            path2, target2 = self.img_path[randNumber], self.labels[randNumber]
            pic2 = Image.open(path2).convert("RGB")

            img_transformed = self.transform(pic2.copy())
            img_list.append(img_transformed)

            img_transformed = self.transform(pic2.copy())
            img_trans_list.append(img_transformed)

            img_transformed = self.transform(pic.copy())
            img_trans_list.append(img_transformed)

        data = stack(img_list,0)
        data_transform = stack(img_trans_list,0)

        return data, data_transform
    
    def get_cls_num_list(self):
        cls_num_list = [[0 for _ in range(1)] for _ in range(self.cls_num)]
        for each_label in self.labels:
                cls_num_list[each_label][0]=cls_num_list[each_label][0] + 1
        
        cls_list = list()
        for each_label in cls_num_list:
            cls_list.append(each_label[0])
        return cls_list
