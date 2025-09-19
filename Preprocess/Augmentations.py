# These are the transformations
import torchvision
import torchvision.transforms as transforms
from configparser import ConfigParser

from torchvision.transforms.functional import normalize
#import config

def GetNormalize(dataset):
  normalize = None
  if(dataset == "Cifar10" or dataset == "Cifar10_lt"):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                      std=[0.247, 0.243, 0.262]) # CIFAR10
  if(dataset == "Imagenet" or dataset == "Stl10" or dataset == "TinyImagenet"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]) #Imagenet, #STL10, #Tiny-Imagenet, #cub200-2011
  if(dataset == "Cifar100"):
    normalize = transforms.Normalize(mean = [0.5071, 0.4867, 0.4408], 
                                        std = [0.2675, 0.2565, 0.2761]) #CIFAR100
  if(dataset == "Cub200"):
    normalize = transforms.Normalize(mean = [0.485, 0.499, 0.432],
                                        std = [0.228, 0.227, 0.266]) #Cub200
  if(dataset == "Aircrafts"):
    normalize = transforms.Normalize(mean = [0.489, 0.487, 0.455],
                                        std = [0.246, 0.242, 0.268]) #Aircrafts
  if(dataset == "Cars"):
    normalize = transforms.Normalize(mean = [0.470, 0.460, 0.454],
                                        std = [0.267, 0.265, 0.270]) #Cars
  if(dataset == "Pets"):
    normalize = transforms.Normalize(mean = [0.485, 0.449, 0.432],
                                        std = [0.229, 0.224, 0.225]) #Pets
  if(normalize == None):
    raise Exception("Datasets are as - cifar-10, Imagenet, cifar-100, TinyImagenet, STL-10, Imagenet")
  
  return normalize
  
class model_transforms:
  def __init__(self,dataset,image_size):
    self.dataset = dataset
    self.image_size = image_size

  def GetTransform(self):
    '''
    config_object = ConfigParser()
    config_object.read("Config/con.dat")

    auginfo = config_object["augmentations"]
    '''

    normalize = GetNormalize(self.dataset)

    #color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
    #                                      saturation=0.4, hue=0.1)

    #rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    #rnd_gray = transforms.RandomGrayscale(p=0.2)
    rnd_rcrop = transforms.RandomResizedCrop(size=self.image_size, scale=(0.08, 1),
            interpolation=transforms.InterpolationMode.BILINEAR)

    resize_image = torchvision.transforms.Resize(size=(self.image_size,self.image_size))
    rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)

    #augmented_train_transform = transforms.Compose([resize_image,rnd_rcrop, rnd_hflip,
    #                                  rnd_color_jitter, rnd_gray,
    #                                  transforms.ToTensor(), normalize])
    augmented_train_transform = transforms.Compose([rnd_rcrop, rnd_hflip,
                                      transforms.ToTensor(), normalize])

    counterpart_train_transform = transforms.Compose([resize_image,transforms.ToTensor(),normalize])
    #counterpart_train_transform = transforms.Compose([transforms.ToTensor(),normalize])

    return counterpart_train_transform, augmented_train_transform

