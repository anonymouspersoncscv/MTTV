from torch import nn, optim
import pytorch_lightning as pl
import torch
import torchvision
from Loss.loss import xent_loss
from Utils.metric.similarity_mean import Positive_Negative_Mean
import torchmetrics
import os
import utils
from Pretraining.Knn_Monitor import Knn_Monitor
from copy import deepcopy
import math

class SimclrModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()
        #self.automatic_optimization = False
        self.net = utils.GetBackbone(config.backbone.name, config.dataset.name)
        
        self.head = nn.Sequential(
            nn.Linear(config.backbone.feature_size, config.model.hidden_size),
            nn.ReLU(),
            nn.Linear(config.model.hidden_size, config.model.projection_size),
        )
        self.lr = config.training.lr
        self.max_epochs = config.training.max_epochs
        self.weight_decay = config.training.weight_decay
        self.warmup_lr = config.training.warmup_lr
        self.warmup_epochs = config.training.warmup_epochs
        self.batch_size = config.training.batch_size
        self.image_size = config.dataset.image_size
        self.save_path = config.dataset.save_path
        self.dataset = config.dataset.name

        self.checkpoint_dir = utils.GetCheckpointDir(config, train_mode="pretrain")
        self.model_name = config.model.name
        self.temperature = config.model.temperature
        self.backbone = config.backbone.name
        self.gpu = config.training.gpu

        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = config.dataset.num_classes)
        self.loss_metric = torchmetrics.MeanMetric()
        self.outputs = []
        self.val_outputs = []
        self.alignment = 0.0
        self.uniformity = 0.0
        self.pos_mean = 0.0
        self.neg_mean = 0.0
        self.total = 0
        # Initialize lists to store layer names and mean weights per epoch
        self.knn_monitor = Knn_Monitor(config)
        self.layer_names = []
        self.mean_weights_per_epoch = {name: [] for name, module in self.net.named_modules() if isinstance(module, torch.nn.Conv2d)}
        self.conv_layer_configs = []
        self.loss_value = []

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x
   
    def training_step(self, batch, batch_idx): 
        
        train_x, _ = batch
        self.optimizer.zero_grad()
        self.embeddings = self.forward(train_x)
        
        loss = xent_loss(self.embeddings)

        self.loss_metric.update(loss)
        
        size = len(self.embeddings) // 2
        positive_pairs = torch.tensor([[i, i+size] for i in range(size)]).to(self.global_rank)
        alignment = utils.alignment_loss(embeddings=self.embeddings, positive_pairs=positive_pairs)
        uniformity = utils.uniformity_loss(embeddings=self.embeddings)

        
        pos_mean, neg_mean = Positive_Negative_Mean(x = self.embeddings, device = self.global_rank)
        self.pos_mean += pos_mean * len(train_x)
        self.neg_mean += neg_mean * len(train_x)
        self.alignment += alignment * len(train_x)
        self.uniformity += uniformity * len(train_x)
        self.total += len(train_x)
        
        self.log_dict(
            {
                'train_loss': self.loss_metric.compute(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'step': float(self.current_epoch),

            },
            on_step = True,
            on_epoch = False,
            prog_bar = True,
            sync_dist=True,
        )
        
        if(batch_idx % 100 == 0):
            y1 = train_x[:4]
            grid_x = torchvision.utils.make_grid(y1.view(-1, 3, self.image_size, self.image_size))
            self.logger.experiment.add_image("Cifar_view1", grid_x, self.current_epoch)
        
        return loss

    def on_train_epoch_end(self):
        #pos_mean, neg_mean = Positive_Negative_Mean(x = self.embeddings, device = self.global_rank, batch_size = self.batch_size)
        self.log_dict(
            {
                'pos_mean': self.pos_mean/self.total,
                'neg_mean': self.neg_mean/self.total,
                'alignment': self.alignment/self.total,
                'uniformity': self.uniformity/self.total,
                'step': float(self.current_epoch),
                #'MeanWeight': sum(weights) / len(weights),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.loss_value.append(self.loss_metric.compute().cpu().numpy())
        self.loss_metric.reset()
        self.accuracy.reset()
        self.alignment = 0.0
        self.uniformity = 0.0
        self.pos_mean = 0.0
        self.neg_mean = 0.0
        self.total = 0
        
        #save model .. 
        if((self.current_epoch + 1) % 100 == 0):
            save_path = os.path.join(self.save_path, self.checkpoint_dir)

            if(not os.path.exists(save_path)):
                os.makedirs(save_path)
                print("Path created...")
            file_path = os.path.join(save_path, "model" + str(self.current_epoch + 1) + ".tar")
            self.Save(file_path)
        
        if((self.current_epoch + 1) % 10 == 0):
            top1 = self.knn_monitor.test(deepcopy(self.net), dataset_name=self.dataset)

            self.log_dict(
                {
                    'Knn Top-1': top1,
                    'step': self.current_epoch,
                    #'Knn Top-5': top5,
                },
                on_epoch = True,
                prog_bar = True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        self.optimizer = optim.SGD([{'params': self.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': self.weight_decay}])
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup_cosine_decay_leraning_rate)
        
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",  # Step the scheduler every epoch
                "frequency": 1,      # Step every 1 epoch
                "monitor": "Knn Top-1",  # Monitor value for advanced schedulers
            },
        }
        
    # warmup + cosine decay
    def warmup_cosine_decay_leraning_rate(self, current_epoch):
        if current_epoch < self.warmup_epochs:
            # Linear warmup: Increase from warmup_lr to base_lr
            return self.warmup_lr + (self.lr - self.warmup_lr) * (current_epoch / self.warmup_epochs) / self.lr
        else:
            # Cosine decay
            decay_epochs = self.epochs - self.warmup_epochs
            decay_epoch = current_epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_epoch / decay_epochs))
            return cosine_decay
    
    """
    Function to save Our Model
    """
     
    def Save(self, model_path):
        
        model_state_dict = self.net.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        torch.save({"model": model_state_dict,
                    "optimizer": optimizer_state_dict},
                    model_path)
        print("model saved succesfully ..")

        
        
