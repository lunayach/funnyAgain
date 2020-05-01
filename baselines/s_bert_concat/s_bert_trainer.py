"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from pytorch_lightning import Trainer, LightningModule
from argparse import ArgumentParser
from s_bert_concat import S_BERT_Regression
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger


import os


def main(hparams):
    # init module

    logger = TestTubeLogger(
        save_dir='./savepath',
        version=1  # An existing version with a saved checkpoint
    )

    if hparams.train:

        model = S_BERT_Regression(hparams)
        checkpoint_callback = ModelCheckpoint(
            filepath=os.getcwd(),
            verbose=True,
            mode='min',
            prefix=hparams.pre,
            monitor = 'avg_val_loss'
        )

        trainer = Trainer(
            max_epochs=hparams.max_nb_epochs,
            gpus=hparams.gpus,
            nb_gpu_nodes=hparams.nodes,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=False,
            fast_dev_run=False,
            overfit_pct=0.0,
            logger = logger
        )

        trainer.fit(model)

    else:
        model = S_BERT_Regression.load_from_checkpoint('edit-unedit-diff_ckpt_epoch_12.ckpt')
        trainer = Trainer()
        trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.add_argument('--pre', type=str, default='')


    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = S_BERT_Regression.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
