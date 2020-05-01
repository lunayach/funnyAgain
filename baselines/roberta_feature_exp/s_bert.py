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
        # self.hparams = hparams
        # self.l1 = torch.nn.Linear(768*2, 256)
        # self.l2 = torch.nn.Linear(256, 128)
        # self.l3 = torch.nn.Linear(128, 1)
        # self.dropout = torch.nn.Dropout(p=0.2)

        self.hparams = hparams
        self.l1 = torch.nn.Linear(768 * 3, 768)
        self.l2 = torch.nn.Linear(768, 256)
        self.l3 = torch.nn.Linear(256, 1)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        f1 = self.dropout(torch.relu(self.l1(x.view(x.size(0), -1))))
        f2 = torch.relu(self.l2(f1))
        out = self.l3(f2)

        # f1 = torch.relu(self.l1(x.view(x.size(0), -1)))
        # out = self.l2(f1)

        return out

    def training_step(self, batch, batch_idx):
        # id, edited, unedited, y = batch
        # y_hat = self.forward(edited)

        # id, edited, unedited, y = batch
        # difference = edited - unedited
        # x = torch.cat((edited, difference), 1)
        # y_hat = self.forward(x)

        id, edited, unedited, y = batch
        difference = edited - unedited
        x = torch.cat((edited, unedited, difference), 1)
        y_hat = self.forward(x)

        return {'loss': F.mse_loss(y_hat.squeeze(), y)}

    def validation_step(self, batch, batch_idx):
        # id, edited, unedited, y = batch
        # y_hat = self.forward(edited)

        # id, edited, unedited, y = batch
        # difference = edited - unedited
        # x = torch.cat((edited, difference), 1)
        # y_hat = self.forward(x)

        id, edited, unedited, y = batch
        difference = edited - unedited
        x = torch.cat((edited, unedited, difference), 1)
        y_hat = self.forward(x)

        return {'val_loss': F.mse_loss(y_hat.squeeze(), y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        # id, edited, unedited, score = batch
        # y_hat = self.forward(edited)

        # id, edited, unedited, y = batch
        # difference = edited - unedited
        # x = torch.cat((edited, difference), 1)
        # y_hat = self.forward(x)

        # id, edited, unedited, score = batch
        # difference = edited - unedited
        # x = torch.cat((edited, unedited, difference), 1)
        # y_hat = self.forward(x)

        id, edited1, unedited1, edited2, unedited2 = batch
        difference1 = edited1 - unedited1
        difference2 = edited2 - unedited2

        x1 = torch.cat((edited1, unedited1, difference1), 1)
        y_hat1 = self.forward(x1)

        x2 = torch.cat((edited2, unedited2, difference2), 1)
        y_hat2 = self.forward(x2)

        result = y_hat1 > y_hat2
        r = [int(re[0]) for re in result]
        preds = []
        for rr in r:
            if rr == 1:
                preds.append(1)
            else:
                preds.append(2)

        return {'pred': preds, 'id': id}

    def test_end(self, outputs):
        all_preds = []
        all_ids = []
        for x in outputs:
            all_preds += x['pred']
            all_ids += list(x['id'])

        all_preds = [int(ap) for ap in all_preds]
        all_ids = [str(ai) for ai in all_ids]

        df = pd.DataFrame(data={'id': all_ids, 'pred': all_preds})
        df.to_csv("./task-2-output.csv", sep=',', index=False)
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
        return DataLoader(S_bert_not_finetuend_Data(val=True), batch_size=self.hparams.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(S_bert_not_finetuend_Data(val=True), batch_size=self.hparams.batch_size)

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

    def __init__(self, train=False, val=False, test=False):
        super(S_bert_not_finetuend_Data, self).__init__()
        self.split = ''
        if train:
            self.split = 'train'
        elif val:
            self.split = 'val'
        elif test:
            self.split = 'test'

        # data_file = '/home/speed/PycharmProjects/transformers/examples/Roberta_finetuned_i2.hdf5'
        data_file = '/home/speed/PycharmProjects/transformers/examples/Roberta_finetuned_task-2_v1.hdf5'
        id_list = open('/home/speed/PycharmProjects/transformers/examples/task-2_ids.txt', 'r')
        self.IDS = id_list.readlines()
        id_list.close()

        self.data = h5py.File(data_file, "r")

    def __getitem__(self, index):

        # dp = self.data[self.split][str(index)]
        dp = self.data['test'][str(index)]

        if not self.split == 'test':
            # return int(np.array(dp['ID'])), np.array(dp['Edited']), np.array(dp['Unedited']), np.array(dp['Score'])
            return str(self.IDS[index]).strip('\n'), np.array(dp['Edited1']), np.array(dp['Unedited1']), np.array(
                dp['Edited2']), np.array(dp['Unedited2'])

        else:
            return int(np.array(dp['ID'])), np.array(dp['Edited']), np.array(dp['Unedited'])

    def __len__(self):
        if self.split == 'train':
            return 9652
        elif self.split == 'val':
            # return 2419
            return 2355
        else:
            return 3024
