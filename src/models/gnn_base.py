import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch
import numpy as np

from .utils import load_dataset
from sklearn.metrics import roc_auc_score


class GNNBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Assign hyperparameters
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering

        if self.trainset is None:
            print("Setting up dataset")

            self.trainset, self.valset, self.testset = load_dataset(
                    input_dir=self.hparams["input_dir"],
                    data_split=self.hparams["data_split"],
            )

        if (
            (self.trainer)
            and ("logger" in self.trainer.__dict__.keys())
            and ("_experiment" in self.logger.__dict__.keys())
        ):
            self.logger.experiment.define_metric("val_loss", summary="min")
            self.logger.experiment.define_metric("auc", summary="max")
            self.logger.experiment.log({"auc": 0})

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(
                self.trainset, batch_size=self.hparams["batch_size"], num_workers=4
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=1, num_workers=0
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(
                self.testset, batch_size=1, num_workers=0
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def get_input_data(self, batch):

        input_data = batch.x
        input_data[input_data != input_data] = 0

        return input_data
    
    def get_loss(self, output, truth, weight):

        positive_loss = F.binary_cross_entropy_with_logits(
            output[truth], torch.ones(truth.sum()).to(self.device)
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            output[~truth], torch.zeros((~truth).sum()).to(self.device)
        )

        loss = positive_loss*weight + negative_loss

        return loss

    def training_step(self, batch, batch_idx):

        truth = batch[self.hparams["truth_key"]].bool()
        edges = batch.edge_index

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~truth.bool()).sum() / truth.sum())
        )

        input_data = self.get_input_data(batch)
        output = self(input_data, edges).squeeze()

        loss = self.get_loss(output, truth, weight)

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def log_metrics(self, output, batch, loss, log):

        preds = torch.sigmoid(output) > self.hparams["edge_cut"]

        # Positives
        edge_positive = preds.sum().float()

        # Signal true & signal tp
        sig_truth = batch[self.hparams["truth_key"]]
        sig_true = sig_truth.sum().float()
        sig_true_positive = (sig_truth.bool() & preds).sum().float()
        sig_auc = roc_auc_score(
            sig_truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
        )

        # Eff, pur, auc
        sig_eff = sig_true_positive / sig_true
        sig_pur = sig_true_positive / edge_positive

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {
                    "val_loss": loss,
                    "current_lr": current_lr,
                    "eff": sig_eff,
                    "pur": sig_pur,
                    "auc": sig_auc,
                },
                sync_dist=True,
            )

        return preds

    def shared_evaluation(self, batch, batch_idx, log=True):

        truth = batch[self.hparams["truth_key"]].bool()
        edges = batch.edge_index
        
        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~truth.bool()).sum() / truth.sum())
        )
        
        input_data = self.get_input_data(batch)
        output = self(input_data, edges).squeeze()

        loss = self.get_loss(output, truth, weight)

        preds = self.log_metrics(output, batch, loss, log)

        return {"loss": loss, "preds": preds, "score": torch.sigmoid(output)}

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx)

        return outputs["loss"]

    def test_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=False)

        return outputs

    def test_step_end(self, output_results):

        print("Step:", output_results)

    def test_epoch_end(self, outputs):

        print("Epoch:", outputs)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()