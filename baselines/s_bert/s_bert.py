import os
import torch, numpy as np, pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
import h5py

import pytorch_lightning as pl


class S_BERT_Regression(pl.LightningModule):

    def __init__(self, hparams):
        super(S_BERT_Regression, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.l1 = torch.nn.Linear(768, 256)
        self.l2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        f1 = torch.relu(self.l1(x.view(x.size(0), -1)))
        out = self.l2(f1)

        return out

    def training_step(self, batch, batch_idx):
        id, x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.mse_loss(y_hat.squeeze(), y)}

    def validation_step(self, batch, batch_idx):
        id, x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.mse_loss(y_hat.squeeze(), y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        id, x = batch
        y_hat = self.forward(x)
        return {'pred': y_hat, 'id': id}

    def test_end(self, outputs):
        all_preds = []
        all_ids = []
        for x in outputs:
            all_preds += list(x['pred'])
            all_ids += list(x['id'])

        all_preds = [float(ap) for ap in all_preds]
        all_ids = [int(ai) for ai in all_ids]

        df = pd.DataFrame(data={'id': all_ids, 'pred': all_preds})
        df.to_csv("./task-1-output.csv", sep=',', index=False)
        return {'all_preds': all_preds, 'all_ids': all_ids}

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
        # TODO explicitly added train=True for the val part
        return DataLoader(S_bert_not_finetuend_Data(train=True), batch_size=self.hparams.batch_size)

    @pl.data_loader
    def test_dataloader(self):
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


class S_bert_not_finetuend_Data(Dataset):

    def __init__(self, train=True):
        super(S_bert_not_finetuend_Data, self).__init__()
        self.train = train
        self.split = 'train'

        if self.train:
            data_file = '../../data/s_bert_not_finetuned_TRAIN.hdf5'
        else:
            data_file = '../../data/s_bert_not_finetuned_DEV.hdf5'
            self.split = 'val'

        self.data = h5py.File(data_file, "r")

    def __getitem__(self, index):

        dp = self.data[self.split][str(index)]

        if self.train:
            return int(np.array(dp['ID'])), np.array(dp['Embedding']), np.array(dp['Score'])
        else:
            return int(np.array(dp['ID'])), np.array(dp['Embedding'])

    def __len__(self):
        # todo remove this hardcoding!!
        if self.train:
            return 9652
        else:
            return 2419
