# MTTV (More Than Two View)
This is an official implementations of the paper "Maximally Useful and Minimally Redundant: The Key to Self Supervised Learning for Imbalanced Data".

## Requirements

    Python              - 3.10.12 
    Tensorboard         - 2.13.0  
    Pytorch             - 1.12.0+cu116 
    Pytorch-lightning   - 1.9.5 

## config settings
    config

        config.yaml

        backbone
            resnet18.yaml
            resnet50.yaml

        dataset
            Cifar100.yaml
            Cifar10.yaml
            Imagenet.yaml

        model
            MTTV.yaml
            MocoV2.yaml
            Simclr.yaml

        post_training
            linear_evaluation.yaml
            transfer_learning.yaml

        training
            defauly.yaml
            pretraining.yaml

## How to Run

### (Pretraining & linear_evaluation)
    
    python3 main.py dataset.data_dir="path_to_dataset" dataset.save_path="path_to_save_model_on_each_nth_epoch" 
    Note - Default setting are - Dataset - Cifar100, Model - MTTV, Backbone - resnet18
 
#### or 

    As per the config files hierarchy use command line arguments to use more parameters

#### or

    Make changes in respective config file and then run - python3 main.py

### Transfer_Learning

    Run python3.main
    Note - set transer_learning = True in transfer_learning.yaml
    It works as - source dataset = transfer_learning.transfer_from, target_dataset = dataset (from config.yaml) 

## To access the tensorboard logs

    tensorboard --logdir results/pretrain_logs/

