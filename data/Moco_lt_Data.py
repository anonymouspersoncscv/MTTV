import torch
import torchvision
from PIL import Image
import os
import numpy as np
import random
"""
Function to open an image and convert it into RGB
"""
def default_loader(path):
    return Image.open(path).convert('RGB')


"""
CIFAR10-LT Dataset class - returns the pair of image 
"""
class MocoDataFromCIFAR10LT(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, imb_type='exp', imb_factor=0.01, rand_number=0, **kwds):
        super().__init__(**kwds)
        #super(OurDataFromCIFAR10LT, self).__init__(root, train, download)
        #self.K = K
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
    
    #For Moco
    
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2, index
        

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


"""
CIFAR100-LT Dataset class - returns the pair of image 
"""

class MocoDataFromCIFAR100LT(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, imb_type='exp', imb_factor=0.01, rand_number=0, **kwds):
        super().__init__(**kwds)
        #super(OurDataFromCIFAR10LT, self).__init__(root, train, download)
        #self.K = K
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
    
    #For Moco
    
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2, index
        

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


"""
Imagenet-LT Dataset class - returns the pair of image 
"""
'''Pareto 2'''

class MocoDataFromImagenetLT(torchvision.datasets.ImageNet):

    def __init__(self, root, txt, transform=None, subset_indices=None):
        self.img_path = []
        self.labels = []
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

    
    def loader(self, path):
        return Image.open(path).convert('RGB')
    
    def __getitem__(self, index):
        if(self.subset_indices is not None):
            real_idx = self.subset_indices[index]
        else:
            real_idx = index
        path, target = self.img_path[real_idx], self.labels[real_idx]
        img = Image.open(path).convert("RGB")
        
        if self.transform is not None:
            img_1 = self.transform(img)
            img_2 = self.transform(img)

        return img_1, img_2, real_idx
    
    def get_cls_num_list(self):
        cls_num_list = [[0 for _ in range(1)] for _ in range(self.cls_num)]
        for each_label in self.labels:
                cls_num_list[each_label][0]=cls_num_list[each_label][0] + 1
        
        cls_list = list()
        for each_label in cls_num_list:
            cls_list.append(each_label[0])
        return cls_list
