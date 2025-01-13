import warnings
import os
import json
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from src.dataset import VQCodeDataset, get_dataset_filelist
from src.encoder import Encoder
from src.quantizer.fsq import FiniteScalarQuantize
from src.utils import count_parameters, AttrDict


class VQCodeLightningModule(LightningModule):
    def __init__(self, h, training_filelist, validation_filelist):
        super().__init__()
        self.save_hyperparameters(h)
        self.h = h

        # Models
        self.encoder = Encoder(h)
        self.quantizer = FiniteScalarQuantize(h.upsample_initial_channel, h.fsq_levels)
        self.code_projector = torch.nn.Linear(h.upsample_initial_channel, h.hubert_dim)

        # Criterion
        self.encoder_criterion = torch.nn.CosineEmbeddingLoss()

        # Datasets
        self.trainset = VQCodeDataset(training_filelist, h.segment_size, h.code_hop_size,
                                       h.sampling_rate)
        self.validset = VQCodeDataset(validation_filelist, h.segment_size, h.code_hop_size,
                                       h.sampling_rate, False)

    def forward(self, x):
        encoder_buf = self.encoder.init_buffers(x.shape[0], x.device)
        y_s_hat, encoder_buf = self.encoder(x, encoder_buf)
        quantized = self.quantizer(y_s_hat)
        projected = self.code_projector(quantized.z.mT)
        return projected

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.unsqueeze(1)
        projected = self(x)
        
        bs = y.shape[0]
        projected_reshaped = projected.permute(0, 2, 1).contiguous().view(bs, -1)
        y_reshaped = y.contiguous().view(bs, -1)
        encoder_loss = self.encoder_criterion(projected_reshaped, y_reshaped, torch.ones(bs).to(self.device))

        self.log("train_loss", encoder_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return encoder_loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.unsqueeze(1)
        projected = self(x)

        bs = y.shape[0]
        projected_reshaped = projected.permute(0, 2, 1).contiguous().view(bs, -1)
        y_reshaped = y.contiguous().view(bs, -1)
        accuracy = F.cosine_similarity(projected_reshaped, y_reshaped, dim=1)
        accuracy_mean = torch.mean(accuracy.float())

        self.log("val_accuracy", accuracy_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return accuracy_mean

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(itertools.chain(self.encoder.parameters(), self.quantizer.parameters(),
                                                      self.code_projector.parameters()), 
                                      self.h.learning_rate, betas=[self.h.adam_b1, self.h.adam_b2])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.h.lr_decay)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.h.batch_size, num_workers=self.h.num_workers,
                          shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.h.batch_size, num_workers=self.h.num_workers,
                          shuffle=False, pin_memory=True, drop_last=False)
