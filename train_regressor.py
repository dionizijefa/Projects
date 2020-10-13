#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:54:12 2020

@author: dionizije
"""

import dataclasses
import shutil
from pathlib import Path
from pprint import pformat
from time import time
from typing import List, Dict, Optional
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import DataLoader
from regressor import GAT
from torch.optim import Adam
from torch.utils.data.sampler import WeightedRandomSampler
from create_pytorch_graphs import graphs
from pytorch_lightning.metrics.functional import regression as pl_regression
from sklearn.metrics import r2_score

root = str(Path(__file__).resolve().parents[1].absolute())
cell_name = 'CVCL_0031'
split_number = "0"
split_root = "{}/data/processed/uncommon_splits/{}/{}/".format(
    root,
    cell_name, 
    split_number
    )
torch.autograd.set_detect_anomaly(False)

@dataclasses.dataclass(frozen=True)
class Conf:
    gpus: int = 2
    seed: int = 0
    use_16bit: bool = False

    lr: float = 1e-4
    batch_size: int = 256
    epochs: int = 150
    
    ckpt_path: Optional[str] = None
    reduce_lr: Optional[bool] = False

    save_dir: str = str(root + "/models/GAT/experiments/uncommon_splits/")

    def __post_init__(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def to_hparams(self) -> Dict:
        excludes = [
            'ckpt_path',
            'reduce_lr',
        ]
        return {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if k not in excludes
        }

    def __str__(self):
        return pformat(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class Metrics:
    lr: float
    loss: float
    mae: float
    rmse: float
    r2: float


class Net(pl.LightningModule):
    def __init__(self, hparams, reduce_lr: Optional[bool] = False):
        super().__init__()
        self.hparams = hparams
        self.reduce_lr = reduce_lr
        self.model = GAT()

        self.best: float = float('inf')


    def forward(self, x, edge_index, edge_attr, batch):
        out = self.model(x, edge_index, edge_attr, batch)
        return out

    def training_step(self, batch, batch_idx):
        result = self.__step(batch, batch_idx, prefix='train')
        return {
            'loss': result['train_loss'],
            **result,
        }

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, prefix='val')
    
    def test_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, prefix='test')

    def training_epoch_end(self, outputs):
        metrics = self.__collect_metrics(outputs, 'train')
        self.__log(metrics, 'train')

        return {}

    def validation_epoch_end(self, outputs):
        metrics = self.__collect_metrics(outputs, 'val')
        self.__log(metrics, 'val')

        if metrics.loss < self.best:
            self.best = metrics.loss

        return {
            'progress_bar': {
                'val_loss': metrics.loss,
                'best': self.best,
            },
        }
    
    def test_epoch_end(self, outputs):
        metrics = self.__collect_metrics(outputs, 'test')
        self.__log(metrics, 'test')

        if metrics.loss < self.best:
            self.best = metrics.loss

        return {
            'progress_bar': {
                'test_loss': metrics.loss,
                'rmse': metrics.rmse,
                'mae': metrics.mae,
                'R^2': metrics.r2
            },
        }

    def __step(self, batch, batch_idx, prefix: str):
        x = batch.x
        edge_index = batch.edge_index
        mini_batch = batch.batch
        edge_attr = batch.edge_attr
        y_hat = self.forward(x, 
                             edge_index, 
                             edge_attr,
                             mini_batch).squeeze(-1)

        loss = pl_regression.mse(y_hat, batch.y)
        rmse = torch.sqrt(loss)
        mae = pl_regression.mae(y_hat, batch.y)
        predictions = y_hat.detach().cpu().numpy()
        r2 = r2_score(predictions, batch.y.cpu())


        return {
            f'{prefix}_loss': loss,
            f'{prefix}_size': len(y_hat),
            f'{prefix}_rmse': rmse,
            f'{prefix}_mae': mae,
            f'{prefix}_r2': r2,
            }

    def __collect_metrics(self, outputs: List[Dict], prefix: str) -> Metrics:
        loss = 0
        total_size = 0
        rmse = 0
        mae = 0
        r2 = 0

        for o in outputs:
            loss += o[f'{prefix}_loss'] * o[f'{prefix}_size']
            total_size += o[f'{prefix}_size']
            rmse += o[f'{prefix}_rmse'] * o[f'{prefix}_size']
            mae += o[f'{prefix}_mae'] * o[f'{prefix}_size']
            r2 += o[f'{prefix}_r2'] * o[f'{prefix}_size']

        # noinspection PyTypeChecker
        return Metrics(
            lr=self.trainer.optimizers[0].param_groups[0]['lr'],
            loss=loss/total_size,
            rmse=rmse/total_size,
            mae=mae/total_size,
            r2=r2/total_size,
            )

    def __log(self, metrics: Metrics, prefix: str):
        if self.global_step > 0:
            self.logger.experiment.add_scalar('lr',
                                              metrics.lr, 
                                              self.current_epoch)
            self.logger.experiment.add_scalars(f'loss', 
                                               {prefix: metrics.loss}, 
                                               self.current_epoch)
            self.logger.experiment.add_scalars(f'rmse', 
                                               {prefix: metrics.rmse}, 
                                               self.current_epoch)

            self.logger.experiment.add_scalars(f'mae', 
                                               {prefix: metrics.mae}, 
                                               self.current_epoch)
            
            self.logger.experiment.add_scalars(f'R^2', 
                                               {prefix: metrics.r2}, 
                                               self.current_epoch)
            
            
            #self.experiment.add_graph(self.model)
            
    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr= self.hp.lr,
            amsgrad = True
        )

        if self.reduce_lr is False:
            return [opt]

        sched = {
            'scheduler': ReduceLROnPlateau(
                opt,
                mode='min',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
            ),
            'interval': 'step',
            'monitor': 'val_ap'
        }
        return [opt], [sched]


    def train_dataloader(self):
        train_dataset = graphs(split_root, subset='nci_train').shuffle()
        class_sample_count = train_dataset.balance_sampler()
        weights = 1 / torch.Tensor(class_sample_count)
        samples_weights = weights[train_dataset.targets]
        sampler = WeightedRandomSampler(samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)

        return DataLoader(train_dataset,
                          self.hp.batch_size,
                          sampler=sampler,
                          num_workers=6, drop_last=True)

    def val_dataloader(self):
        val_dataset = graphs(split_root, subset='nci_valid')
        return DataLoader(val_dataset,
                          self.hp.batch_size,
                          shuffle=False,
                          num_workers=6)

    def test_dataloader(self):
        test_dataset = graphs(split_root, subset='nci_test')

        return DataLoader(test_dataset,
                          self.hp.batch_size,
                          shuffle=False,
                          num_workers=6)

    @property
    def hp(self) -> Conf:
        return Conf(**self.hparams)

    @property
    def steps_per_epoch(self) -> int:
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        return len(self.train_dataloader())    
    
def main(conf: Conf):
    model = Net(
        conf.to_hparams(),
        reduce_lr=conf.reduce_lr,
    )
    
    logger = TensorBoardLogger(
        conf.save_dir,
        name='{}'.format(cell_name),
        version='split_{}_seed_{}_{}'.format(split_number, 
                                             conf.seed,
                                             str(int(time())))

    )

    # Copy this script to log_dir
    log_dir = Path(logger.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(Path(__file__), log_dir)
    shutil.copy(Path(root + "/src/train_regressor.py"), log_dir)
    shutil.copy(Path(root + "/src/create_pytorch_graphs.py"), log_dir)
    shutil.copy(Path(root + "/src/regressor.py"), log_dir)
    shutil.copy(Path(root + "/src/train_test_split.py"), log_dir)

    trainer = pl.Trainer(
        max_epochs=conf.epochs,
        gpus=[0],
        logger=logger,
        resume_from_checkpoint=conf.ckpt_path, #load from checkpoint instead of resume
        weights_summary='top',
        deterministic = True,
        auto_lr_find = False,
        #precision = 16
        )
    trainer.fit(model)
    trainer.test()

if __name__ == '__main__':
    print('Training GIN for drug response prediction')

    main(Conf(

        lr=1e-4,
        batch_size=256,
        epochs=150,

        #ckpt_path=str("/home/dfa/MPNN/models/DimeNet_one_cell/experiments/version/1596100807/checkpoints/epoch=41.ckpt"),

        #reduce_lr=True
    ))
