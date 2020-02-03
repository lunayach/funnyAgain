import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
import h5py

import pytorch_lightning as pl

class S_bert_not_finetuend_Data(Dataset):

      def __init__(self, train=True):
            super(S_bert_not_finetuend_Data, self).__init__()
            self.train = train
            self.split = 'train'

            if self.train:
                  data_file = 'research_seed/data/s_bert_not_finetuned_TRAIN.hdf5'
            else:
                  data_file = 'research_seed/data/s_bert_not_finetuned_DEV.hdf5'
                  self.split = 'val'

            self.data = h5py.File(data_file, "r")

      def __getitem__(self, index):

            dp = self.data[self.split][str(index)]
            if self.train:
                  return dp['ID'], dp['Embedding'], dp['Score']
            else:
                  return dp['ID'], dp['Embedding']

      def __len__(self):
            #todo remove this hardcoding!!
            if self.train:
                  return 9652
            else:
                  return 2419


class S_BERT_Regression(pl.LightningModule):

    def __init__(self, hparams):
        super(S_BERT_Regression, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.l1 = torch.nn.Linear(768, 256)
        self.l2 = torch.nn.Linear(256, 10)


    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        id, x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(S_bert_not_finetuend_Data(train=True), batch_size=self.hparams.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(S_bert_not_finetuend_Data(train=False), batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser

