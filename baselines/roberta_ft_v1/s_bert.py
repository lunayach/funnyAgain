import os
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer
)

import csv, re

import torch, numpy as np, pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
import h5py

import pytorch_lightning as pl


class Roberta_ft(pl.LightningModule):

    def __init__(self, hparams):
        super(Roberta_ft, self).__init__()
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
        config = config_class.from_pretrained('roberta-base', cache_dir=None)
        config.__dict__["output_hidden_states"] = True
        self.model = model_class.from_pretrained(
            '/home/speed/PycharmProjects/transformers/examples/checkpoint-14000-best/pytorch_model_best.bin',
            from_tf=bool(
                ".ckpt" in '/home/speed/PycharmProjects/transformers/examples/checkpoint-14000-best/pytorch_model_best.bin'),
            config=config,
            cache_dir=None,
        )

        self.model.cuda()
        self.model.eval()
        # not the best model...
        self.hparams = hparams
        self.l1 = torch.nn.Linear(768, 256)
        self.l2 = torch.nn.Linear(256, 1)

    def forward(self, edited_input_id, edited_token_type_id, edited_attention_mask):
        print(edited_input_id)
        exit()
        _, hidden = self.model(input_ids=edited_input_id, attention_mask=edited_attention_mask,
                               token_type_ids=edited_token_type_id)
        feature = hidden[-2]
        x = torch.mean(hidden, dim=1).detach().numpy()

        f1 = torch.relu(self.l1(x.view(x.size(0), -1)))
        out = self.l2(f1)

        return out

    def training_step(self, batch, batch_idx):
        id, edited_input_id, edited_token_type_id, edited_attention_mask, unedited_input_id, unedited_token_type_id, unedited_attention_mask, score = batch
        y_hat = self.forward(edited_input_id, edited_token_type_id, edited_attention_mask)
        return {'loss': F.mse_loss(y_hat.squeeze(), score)}

    def validation_step(self, batch, batch_idx):
        id, edited, unedited, score = batch
        y_hat = self.forward(edited)
        return {'val_loss': F.mse_loss(y_hat.squeeze(), score)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        id, edited, unedited = batch
        y_hat = self.forward(edited)
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
        return DataLoader(S_bert_not_finetuend_Data(val=True), batch_size=self.hparams.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(S_bert_not_finetuend_Data(test=True), batch_size=self.hparams.batch_size)

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
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=None)

        file = []
        file1 = []
        if train:
            file = open('/home/speed/PycharmProjects/funnyAgain/data/task-1/train.csv', 'r')
            file1 = open('/home/speed/PycharmProjects/funnyAgain/data/task-1/train_funlines.csv', 'r')
        if val:
            file = open('/home/speed/PycharmProjects/funnyAgain/data/task-1/dev.csv', 'r')
        if test:
            file = open('/home/speed/PycharmProjects/funnyAgain/data/task-1/test.csv', 'r')

        reader = csv.reader(file)
        ids = []
        unedited = []
        edited = []
        scores = []
        edits = []
        replacements = []
        for i, lines in enumerate(reader):
            if i == 0:
                continue
            Id = lines[0]
            line = lines[1]
            edit = lines[2]
            if not test:
                score = lines[4]
                scores.append(score)

            ids.append(Id)
            match = re.search(r'<.*/>', line)
            replacements.append(line[match.start() + 1: match.end() - 2])
            unedited.append(re.sub(r'[</>]', '', line))
            edited.append(re.sub(r'<.*/>', edit, line))
            edits.append(edit)

        # if train:
        #     reader = csv.reader(file1)
        #     fids = []
        #     funedited = []
        #     fedited = []
        #     fscores = []
        #     fedits = []
        #     freplacements = []
        #     for i, lines in enumerate(reader):
        #         if i == 0:
        #             continue
        #         Id = lines[0]
        #         line = lines[1]
        #         edit = lines[2]
        #         score = lines[4]
        #
        #         fids.append(Id)
        #         match = re.search(r'<.*/>', line)
        #         freplacements.append(line[match.start() + 1: match.end() - 2])
        #         funedited.append(re.sub(r'[</>]', '', line))
        #         fedited.append(re.sub(r'<.*/>', edit, line))
        #         fscores.append(score)
        #         fedits.append(edit)
        #
        #     ids = ids + fids
        #     edited = edited + fedited
        #     unedited = unedited + funedited
        #     scores = scores + fscores
        #     edits = edits + fedits

        edited = tokenizer.batch_encode_plus(edited, add_special_tokens=True, max_length=512)
        unedited = tokenizer.batch_encode_plus(unedited, add_special_tokens=True, max_length=512)

        self.ids = ids
        self.unedited = unedited
        self.edited = edited
        self.scores = scores
        self.edits = edits
        self.replacements = replacements

        self.train = train
        self.val = val
        self.test = test

    def __getitem__(self, index):
        if not self.test:
            # print(self.ids[index], self.edited[index], self.unedited[index], self.scores[index])
            # id, edited_input_id, edited_token_type_id, edited_attention_mask, unedited_input_id, unedited_token_type_id, unedited_attention_mask, score
            # print(torch.tensor(self.edited['input_ids'][index], dtype=torch.long).unsqueeze(dim=0).shape)

            # return int(self.ids[index]), torch.tensor(self.edited['input_ids'][index], dtype=torch.long).unsqueeze(dim=0), torch.tensor(
            #     self.edited['token_type_ids'][index], dtype=torch.long).unsqueeze(dim=0), torch.tensor(
            #     self.edited['attention_mask'][index], dtype=torch.long).unsqueeze(dim=0), torch.tensor(
            #     self.unedited['input_ids'][index], dtype=torch.long).unsqueeze(dim=0), torch.tensor(
            #     self.unedited['token_type_ids'][index], dtype=torch.long).unsqueeze(dim=0), torch.tensor(
            #     self.unedited['attention_mask'][index], dtype=torch.long).unsqueeze(dim=0), np.array(self.scores[index])
            # print(type(self.edited['input_ids'][index]))
            # print(self.edited['input_ids'][index])
            # print(self.edited['token_type_ids'][index])
            # print(self.edited['attention_mask'][index])
            # print(self.unedited['input_ids'][index])
            # print(self.unedited['token_type_ids'][index])
            # print(self.unedited['attention_mask'][index])
            # print(np.array(self.scores[index]))
            # exit()

            return int(self.ids[index]), self.edited['input_ids'][index], self.edited['token_type_ids'][index], \
                   self.edited['attention_mask'][index], self.unedited['input_ids'][index], \
                   self.unedited['token_type_ids'][index], self.unedited['attention_mask'][index], np.array(
                self.scores[index])
        else:
            return int(self.ids[index]), torch.tensor(self.edited[index], dtype=torch.long).unsqueeze(
                dim=0), torch.tensor(
                self.unedited[index], dtype=torch.long).unsqueeze(dim=0)

    def __len__(self):
        return len(self.ids)
