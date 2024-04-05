from typing import Any, List

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, AveragePrecision
from torchmetrics.classification.accuracy import Accuracy

from wildfire_forecasting.models.modules.fire_modules import SimpleConvLSTM, SimpleLSTM


def combine_dynamic_static_inputs(dynamic, clc, access_mode):
    clc = None
    assert access_mode in ['spatial', 'temporal', 'spatiotemporal']
    if access_mode == 'spatial':
        dynamic = dynamic.float()
        # static = static.float()
        # input_list = [dynamic, static]
        input_list = [dynamic]
        inputs = torch.cat(input_list, dim=1)
    elif access_mode == 'temporal':
        bsize, timesteps, _ = dynamic.shape
        # static = static.unsqueeze(dim=1)
        # repeat_list = [1 for _ in range(static.dim())]
        # repeat_list[1] = timesteps
        # static = static.repeat(repeat_list)
        # input_list = [dynamic, static]
        input_list = [dynamic]
        if clc is not None:
            clc = clc.unsqueeze(dim=1).repeat(repeat_list)
            input_list.append(clc)
        inputs = torch.cat(input_list, dim=2).float()
    else:
        bsize, timesteps, _, _, _ = dynamic.shape
        # static = static.unsqueeze(dim=1)
        # repeat_list = [1 for _ in range(static.dim())]
        # repeat_list[1] = timesteps
        # static = static.repeat(repeat_list)
        # input_list = [dynamic, static]
        input_list = [dynamic]
        if clc is not None:
            clc = clc.unsqueeze(dim=1).repeat(repeat_list)
            input_list.append(clc)
        inputs = torch.cat(input_list, dim=2).float()
    return inputs


class ConvLSTM_fire_model(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            dynamic_features=None,
            # static_features=None,
            hidden_size: int = 32,
            lstm_layers: int = 1,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005,
            dropout: float = 0.5,
            access_mode='spatiotemporal',
            clc=None
            # clc='vec'
    ):
        super().__init__()
        self.validation_step_outputs = []

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SimpleConvLSTM(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1. - positive_weight, positive_weight]))
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy(task='binary')
        self.train_auc = AUROC(task='binary')
        self.train_auprc = AveragePrecision(task='binary')

        self.val_accuracy = Accuracy(task='binary')
        self.val_auc = AUROC(task='binary')
        self.val_auprc = AveragePrecision(task='binary')

        self.test_accuracy = Accuracy(task='binary')
        self.test_auc = AUROC(task='binary')
        self.test_auprc = AveragePrecision(task='binary')

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        # dynamic, static, clc, y = batch
        dynamic, clc, y = batch
        y = y.long()
        if not self.hparams['clc']:
            clc = None
        # inputs = combine_dynamic_static_inputs(dynamic, static, clc, 'spatiotemporal')
        inputs = combine_dynamic_static_inputs(dynamic, clc, 'spatiotemporal')        
        logits = self.forward(inputs)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        return loss, preds, preds_proba, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'train'

        # log train metrics
        self.train_accuracy.update(preds, targets)
        self.train_auc.update(preds_proba, targets)
        self.train_auprc.update(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     pass

    # def on_train_epoch_end(self, outputs: List[Any]):
    def on_train_epoch_end(self):        
        # `outputs` is a list of dicts returned from `training_step()`
        pass    

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'val'

        # log train metrics
        self.val_accuracy.update(preds, targets)
        self.val_auc.update(preds_proba, targets)
        self.val_auprc.update(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=False)

        self.validation_step_outputs.append(loss)
        self.validation_step_outputs.append(preds)
        self.validation_step_outputs.append(targets)
        self.validation_step_outputs.append(preds_proba)

        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    # def validation_epoch_end(self, outputs: List[Any]):
    #     pass

    # def on_validation_epoch_end(self, outputs: List[Any]):
    def on_validation_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets = self.step(batch)
        phase = 'test'

        # log train metrics
        self.test_accuracy.update(preds, targets)
        self.test_auc.update(preds_proba, targets)
        self.test_auprc.update(preds_proba, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    # def test_epoch_end(self, outputs: List[Any]):
    #     pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_scheduler_step,
                                                       gamma=self.hparams.lr_scheduler_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class LSTM_fire_model(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            dynamic_features=None,
            # static_features=None,
            hidden_size: int = 32,
            lstm_layers: int = 3,
            lr: float = 0.001,
            positive_weight: float = 0.5,
            lr_scheduler_step: int = 10,
            lr_scheduler_gamma: float = 0.1,
            weight_decay: float = 0.0005,
            attention: bool = False,
            dropout: float = 0.5,
            access_mode='temporal',
            clc=None
            # clc='vec'
    ):
        super().__init__()
        self.validation_step_outputs = []

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.attention = attention
        if self.attention:
            self.model = SimpleLSTMAttention(hparams=self.hparams)
        else:
            self.model = SimpleLSTM(hparams=self.hparams)
        self.weight_decay = weight_decay
        # loss function
        self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1 - positive_weight, positive_weight]))
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # Accuracy, AUROC, AUC, ConfusionMatrix
        self.train_accuracy = Accuracy(task='binary')
        self.train_auc = AUROC(task='binary')
        self.train_auprc = AveragePrecision(task='binary')

        self.val_accuracy = Accuracy(task='binary')
        self.val_auc = AUROC(task='binary')
        self.val_auprc = AveragePrecision(task='binary')

        self.test_accuracy = Accuracy(task='binary')
        self.test_auc = AUROC(task='binary')
        self.test_auprc = AveragePrecision(task='binary')

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        # dynamic, static, clc, y = batch
        dynamic, clc, y, path = batch
        y = y.long()
        if not self.hparams['clc']:
            clc = None
        # inputs = combine_dynamic_static_inputs(dynamic, static, clc, 'temporal')
        inputs = combine_dynamic_static_inputs(dynamic, clc, 'temporal')

        logits = self.forward(inputs)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds_proba = torch.exp(logits)[:, 1]
        return loss, preds, preds_proba, y, path

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets, paths = self.step(batch)
        phase = 'train'
        # log train metrics
        self.train_accuracy.update(preds, targets)
        self.train_auc.update(preds_proba, targets)
        self.train_auprc.update(preds_proba, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     pass

    # def on_train_epoch_end(self, outputs: List[Any]):
    def on_train_epoch_end(self):        
        # `outputs` is a list of dicts returned from `training_step()`
        pass    

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets, paths = self.step(batch)
        phase = 'val'
        # log train metrics
        self.val_accuracy.update(preds, targets)
        self.val_auc.update(preds_proba, targets)
        self.val_auprc.update(preds_proba, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=False)

        self.validation_step_outputs.append(loss)
        self.validation_step_outputs.append(preds)
        self.validation_step_outputs.append(targets)
        self.validation_step_outputs.append(preds_proba)

        return {"loss": loss, "preds": preds, "targets": targets, "preds_proba": preds_proba}

    # def validation_epoch_end(self, outputs: List[Any]):
    #     pass

    # def on_validation_epoch_end(self, outputs: List[Any]):
    def on_validation_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, preds_proba, targets, paths = self.step(batch)
        phase = 'test'

        # log train metrics
        self.test_accuracy.update(preds, targets)
        self.test_auc.update(preds_proba, targets)
        self.test_auprc.update(preds_proba, targets)

        # save prediction probs and px info
        targets_np = targets.cpu().numpy()
        preds_proba_np = preds_proba.cpu().numpy()
        preds_np = preds.cpu().numpy()
        loss_np = loss.cpu().numpy()
        paths_str = np.array(paths)
        #
        for i, x in enumerate(paths_str):
            # results = [paths_np[i], preds_proba_np[i]]
            results = [preds_proba_np[i], preds_np[i], targets_np[i]]
            # results = [loss_np[i][0]]
            results = np.array(results)
            opath = str(paths_str[i]).split("/")[-2]+"_"+paths_str[i].split("/")[-1].split(".npy")[0]
            print(opath)

            with open(f"/home/hugosoto/Work/git/ine-on-fire/model/data/valparaiso/npy/temporal/predictions/{opath}.npy", 'wb') as f:
                np.save(f, results)

            # np.savetxt(f"/home/hugosoto/Work/git/ine-on-fire/model/data/valparaiso/npy/temporal/predictions/{opath}.out", results, delimiter=',')
       
        # np.savetxt('/home/hugosoto/Work/git/ine-on-fire/model/data/valparaiso/npy/temporal/predictions/preds_proba_np.out', preds_proba_np, delimiter=',')
        # np.savetxt('/home/hugosoto/Work/git/ine-on-fire/model/data/valparaiso/npy/temporal/predictions/preds_proba_np.out', preds_proba_np, delimiter=',')
        # np.savetxt('/home/hugosoto/Work/git/ine-on-fire/model/data/valparaiso/npy/temporal/predictions/paths_np.out', paths_np, delimiter=',')

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    # def test_epoch_end(self, outputs: List[Any]):
    #     pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """
        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_scheduler_step,
                                                       gamma=self.hparams.lr_scheduler_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
