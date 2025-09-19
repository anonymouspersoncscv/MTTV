import pytorch_lightning as pl
#import config
from Linear_Evaluation.Model_LE import linearlayer_training
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import torch
from models.MocoV2Model import MocoV2Model
from models.MTTVModel import MTTVModel
from models.SimclrModel import SimclrModel

from data.Cifar100.cifar100_dataset_le import Cifar100_DataModule_le
from data.Cifar10.cifar10_dataset_le import Cifar10_DataModule_le
from data.Imagenet.imagenet_dataset_le import Imagenet_DataModule_le

import utils

def Get_Model(model_name):
   
    model_function = model_name + "Model"

    exec(f"generated_model = {model_function}", globals())
    return globals()['generated_model']
    

def Get_Dataset(dataset_name):
    dataset_function = dataset_name + "_DataModule_le"
    exec(f"generated_dataset = {dataset_function}", globals())
    return globals()['generated_dataset']

def Transfer_Learning(config):
     
    result_folder = utils.GetTensorboardDir(config, train_mode="linear_eval")
    
    pretrain_checkpoint_path = utils.GetCheckpointDir(config, train_mode= "pretrain")
    linear_checkpoint_path = utils.GetCheckpointDir(config, train_mode= "linear_eval")
    #linear_checkpoint_path = os.path.join("results", config.dataset.name + "_linear", config.feature.mode, config.imbalance.imb_type)
    logger = TensorBoardLogger(result_folder, name = config.backbone.name)
    
    pretrained_filename = os.path.join(config.dataset.save_path, linear_checkpoint_path, (config.model.name + "ModelLE.ckpt"))
    #pretrained_filename = config.dataset.save_path
    
    checkpoint_callback = ModelCheckpoint(dirpath=pretrained_filename,
                                                 #save_weights_only=True,
                                                 mode = "max",
                                                 monitor='Linear_Evaluation_Acc')
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = linearlayer_training(config)
    generated_dataset = Get_Dataset(config.dataset.name)
    generated_model = Get_Model(config.model.name)
    trained_model = generated_model(config)
    
    #trained_model = utils.GetBackbone(config.backbone.name, config.dataset.name)
    print(config.transfer_learning.dataset + " -----> " + config.dataset.name)
    
    model_path = os.path.join(config.dataset.save_path, pretrain_checkpoint_path)
    checkpoint = os.path.join(model_path, "model" + str(config.transfer_learning.loading_point) + ".tar")
    print(checkpoint)
    checkpoint = torch.load(checkpoint)
    
    if(config.model.name == "MocoV2" or config.model.name == "MTTV"):
        trained_model.encoder_q.load_state_dict(checkpoint["encoder_q"])
        dm = generated_dataset(trained_model.encoder_q[0], config)
    else:
        trained_model.load_state_dict(checkpoint["backbone"])
        dm = generated_dataset(trained_model, config)


    trainer = pl.Trainer(
        default_root_dir=os.path.join(linear_checkpoint_path, (config.model.name + "ModelLE")),
        #strategy='ddp',
        logger = logger,
        accelerator='gpu',
        devices = config.post_training.devices,
        min_epochs = 1,
        max_epochs=config.post_training.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)
        
        
