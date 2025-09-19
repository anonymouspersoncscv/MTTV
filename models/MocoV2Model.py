from torch import nn, optim
import pytorch_lightning as pl
import torch
import torchvision
from Loss.infonce_loss import infonce_loss
from Utils.metric.moco_similarity_mean import Positive_Negative_Mean
import torchmetrics
import os
from copy import deepcopy
import utils
from Pretraining.Knn_Monitor import Knn_Monitor
from copy import deepcopy
import math
import utils

class Projection(nn.Module):
    """ Projector for MoCov2: Copy from SimCLR"""
    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()
        if hidden_dim == None: hidden_dim=in_dim
        self.layer1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, out_dim),
                )
    def forward(self, x):
        x = self.layer1(x)
        return x

class MocoV2Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()
        #self.automatic_optimization = False

        self.net = utils.GetBackbone(config.backbone.name, config.dataset.name)
        self.projection = Projection(config.backbone.feature_size, config.backbone.feature_size, config.model.projection_size)
        self.max_epochs = config.training.max_epochs
        self.gpu = config.training.gpu
         
        self.batch_size = config.training.batch_size
        self.image_size = config.dataset.image_size
        self.save_path = config.dataset.save_path
        self.dataset = config.dataset.name

        self.backbone = config.backbone.name
        self.dim = config.backbone.feature_size
        
        self.warmup_lr = config.training.warmup_lr
        self.warmup_epochs = config.training.warmup_epochs
        self.lr = config.model.lr
        self.temperature = config.model.temperature
        self.momentum = config.model.momentum
        self.weight_decay = config.model.weight_decay
        self.model_name = config.model.name
        self.K = config.model.moco_K
        self.m = config.model.moco_m
        self.T = config.model.moco_T
        self.symmetric = config.model.moco_Symmetric
        self.checkpoint_dir = utils.GetCheckpointDir(config, train_mode="pretrain")
        
        # create the encoders
        self.encoder_q = self.encoder = nn.Sequential(self.net, self.projection)
        
        self.encoder_k = deepcopy(self.encoder_q)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(config.model.projection_size, self.K))
        #self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self._init_encoder_k()

        self.knn_monitor = Knn_Monitor(config)
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
        self.layer_names = []
        self.mean_weights_per_epoch = {name: [] for name, module in self.net.named_modules() if isinstance(module, torch.nn.Conv2d)}
        self.conv_layer_configs = []
        self.loss_value = []

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bz = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % bz == 0
        self.queue[:, ptr:(ptr + bz)] = keys.t()
        ptr = (ptr + bz) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        idx_shuffle = torch.randperm(x.shape[0]).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]
     
    def training_step(self, batch, batch_idx): 
        train_x, train_k = batch
        self.optimizer.zero_grad()
        q = self.encoder_q(train_x)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_encoder_k()
            x_k, idx_unshuffle = self._batch_shuffle_single_gpu(train_k)
            k = self.encoder_k(x_k) 
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
        loss = infonce_loss(q, k, self.queue, self.temperature)
        self._dequeue_and_enqueue(k)

        self.loss_metric.update(loss)
        utils.adjust_learning_rate(self.optimizer,init_lr=self.lr, epoch=self.current_epoch, max_epochs=self.max_epochs)
        
        batch_size = q.shape[0] // 2
        positive_pairs = torch.tensor([[i, i + batch_size] for i in range(batch_size)]).to(self.global_rank)
        
        alignment = utils.alignment_loss(embeddings=q, positive_pairs=positive_pairs)
        uniformity = utils.uniformity_loss(embeddings=q) + utils.uniformity_loss(embeddings=k)
        
        pos_mean, neg_mean = Positive_Negative_Mean(q, k, self.queue)
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
            x1 = train_x[:4]
            grid_x = torchvision.utils.make_grid(x1.view(-1, 3, self.image_size, self.image_size))
            self.logger.experiment.add_image("Cifar_A", grid_x, self.current_epoch)
         
            k2 = train_k[:4]
            grid_k = torchvision.utils.make_grid(k2.view(-1, 3, self.image_size, self.image_size))
            self.logger.experiment.add_image("Cifar_B", grid_k, self.current_epoch)
        

        return loss

    def on_train_epoch_end(self):
         
        self.log_dict(
            {
                'pos_mean': self.pos_mean/self.total,
                'neg_mean': self.neg_mean/self.total,
                'alignment': self.alignment/self.total,
                'uniformity': self.uniformity/self.total,
                'step': float(self.current_epoch),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.loss_value.append(self.loss_metric.compute().cpu().numpy())
        self.loss_metric.reset()
        self.accuracy.reset()
        #self.loss_meter = torch.tensor(0)
        self.alignment = 0.0
        self.uniformity = 0.0
        self.pos_mean = 0.0
        self.neg_mean = 0.0
        self.total = 0

        #save model .. 
        if((self.current_epoch + 1)% 100 == 0):
            save_path = os.path.join(self.save_path, self.checkpoint_dir)

            if(not os.path.exists(save_path)):
                os.makedirs(save_path)
                print("Path created...")
            file_path = os.path.join(save_path, "model" + str(self.current_epoch + 1) + ".tar")
            self.Save(file_path)
             
        if((self.current_epoch + 1)% 10 == 0):
            top1 = self.knn_monitor.test(net = deepcopy(self.encoder_q), dataset_name=self.dataset)

            self.log_dict(
                {
                    'Knn Top-1': top1,
                    'step': float(self.current_epoch),
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
        
        encoder_q_state_dict = self.encoder_q.state_dict()
        encoder_k_state_dict = self.encoder_k.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        torch.save({"encoder_q": encoder_q_state_dict,
                    "encoder_k": encoder_k_state_dict,
                    "optimizer": optimizer_state_dict},
                    model_path)
        print("model saved succesfully ..")

    """
    Function to load Our Model
    """

    def Load(self, file_path, LinearEvaluation = False):
        checkpoint = torch.load(file_path,map_location='cpu')
        if(LinearEvaluation):
            self.encoder_q.net.load_state_dict(checkpoint["encoder_q"]["backbone"])
        else:
            self.encoder_q.load_state_dict(checkpoint["encoder_q"])
            self.encoder_k.load_state_dict(checkpoint["encoder_k"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
