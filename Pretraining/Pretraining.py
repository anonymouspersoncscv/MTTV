from deepspeed.runtime.engine import deepspeed
import pytorch_lightning as pl
import torch
import os

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy,DDPStrategy
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor

from models.MTTVModel import MTTVModel

from models.SimclrModel import SimclrModel

from models.MocoV2Model import MocoV2Model

from data.Imagenet.imagenet_lt_dataset import Imagenet_DataModuleLT
from data.Cifar10.cifar10_lt_dataset import Cifar10_DataModuleLT
from data.Cifar100.cifar100_lt_dataset import Cifar100_DataModuleLT

import utils

def Get_Model(model_name):
    
    model_function = model_name + "Model"
    exec(f"generated_model = {model_function}", globals())
    return globals()['generated_model']
    

def Get_Dataset(dataset_name, imbalance):
    dataset_function = dataset_name + "_DataModule"
    if(imbalance):
        dataset_function = dataset_function + "LT"
    exec(f"generated_dataset = {dataset_function}", globals())
    return globals()['generated_dataset']

#@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def Pretraining(config):
    strategy = DeepSpeedStrategy(logging_batch_size_per_gpu=config.training.batch_size)
    result_folder = utils.GetTensorboardDir(config, train_mode="pretrain")
    logger = TensorBoardLogger(result_folder, name = config.backbone.name)
    profiler = PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler("results/pretrain_logs/profiler0"),
        trace_memory = True,
        schedule = torch.profiler.schedule(skip_first = 10, wait=1, warmup=1, active=20),
    )
    logger_file = logger.log_dir
    checkpoint_path = utils.GetCheckpointDir(config, train_mode="pretrain")
    pretrained_filename = os.path.join(checkpoint_path, (config.model.name + "Model.ckpt"))
    checkpoint_callback = ModelCheckpoint(dirpath=pretrained_filename,
                                          #save_weights_only=True,
                                          every_n_epochs=5,
                                          save_last=True,
                                          mode = 'min',
                                          monitor='train_loss')
    lr_monitor = LearningRateMonitor(logging_interval = 'step')
    generated_model = Get_Model(config.model.name)
    model = generated_model(config)
    generated_dataset = Get_Dataset(config.dataset.name, config.dataset.imbalance)
    dm = generated_dataset(config=config)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path + config.model.name + "Model"),
        #strategy = DDPStrategy(find_unused_parameters=False),
        #profiler = profiler,
        logger = logger,
        accelerator='gpu',
        devices = config.training.devices,
        min_epochs = 1,
        max_epochs=config.training.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],  # , print_time],
        log_every_n_steps=1,
        deterministic=True,
    )
    print(pretrained_filename)
    ''' Uncomment this code to load from last checkpoint by using checkpoint_toload or last.ckpt 
        and comment trainer.fit given below'''
    '''
    if(os.path.exists(pretrained_filename)):
        print("Model Loading..." )
        #saved_backbone = "epoch=" + str(config.training.checkpoint_toload) + "-step=" + str(config.training.checkpoint_toload) +".ckpt"
        saved_backbone = "last-v2.ckpt"
        trainer.fit(model, dm, ckpt_path=os.path.join(pretrained_filename, saved_backbone))
    else:
        pl.seed_everything(42)
        trainer.fit(model, dm)
        #trainer.validate(model, dm)
    '''
    trainer.fit(model, dm)
    #trainer.test(model, dm)
    return model

