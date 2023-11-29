from copy import deepcopy
from itertools import chain, product
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy
from vit_pytorch.simple_vit_1d import SimpleViT

from .basic_arch import EmbeddingNetwork, UnitNormLayer
from .conformer import Conformer
from .data import DataModule, get_per_subj_data, shuffle_together
from .loss import TripletCentroids, TripletLoss, TripletLossHardMining
from .scoring import get_combo_conf_mat
from .utils import PROJECT_PATH


class InsufficientDataError(Exception):
    ...


class DummyIdentity(nn.Module):
    # A null embedding. Has a single (unused) parameter to easily use in the same pl training loop
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x.flatten(1)


class MLPClf(nn.Sequential):
    def __init__(self, input_dim, output_dim):
        layers = [
            nn.Linear(input_dim, input_dim * 2, bias=False),
            nn.BatchNorm1d(input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.05),
            nn.Linear(input_dim * 2, input_dim, bias=False),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.05),
            nn.Linear(input_dim, output_dim),
        ]
        super().__init__(*layers)


class Avg(nn.Module):
    def forward(self, x1, x2, _y1, _y2):
        # Note that vector average is elementwise; thus we don't care
        # if we have a pair of single vectors or a pair of batches
        return (x1 + x2) / 2


class MLPCombine(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layer = nn.Sequential(
            # Input takes 2 feature vectors, and 2 labels (each one-hot with 5 classes)
            nn.Linear(feature_dim * 2 + 5 * 2, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.05),
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.05),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x1, x2, y1, y2):
        y1 = F.one_hot(y1, num_classes=5)
        y2 = F.one_hot(y2, num_classes=5)
        avg = (x1 + x2) / 2
        mlp_out = self.layer(torch.cat((x1, x2, y1, y2), dim=-1))
        return avg + mlp_out


class CombinePairs(nn.Module):
    def __init__(self, combine_fn: nn.Module, normalized_features: bool):
        super().__init__()
        self.normalized_features = normalized_features
        self.combine_fn = combine_fn

    def forward(self, x, y):
        # Expects data and labels from single gestures
        # Labels have the form (direction, modifier)
        # where direction in 0, 1, 2, 3 is active, and 4 is NoDir
        # same for modifier
        device = x.device

        dir_idx = y[:, 1] == 4  # When modifier is NoMod
        mod_idx = y[:, 0] == 4  # When direction is NoDir

        x_dir = x[dir_idx]
        y_dir = y[dir_idx, 0]
        x_mod = x[mod_idx]
        y_mod = y[mod_idx, 1]

        if len(x_dir) * len(x_mod) <= 1:
            raise InsufficientDataError()

        all_x1, all_x2, all_y1, all_y2 = [], [], [], []
        for (x1, y1), (x2, y2) in product(zip(x_dir, y_dir), zip(x_mod, y_mod)):
            all_x1.append(x1)
            all_x2.append(x2)
            all_y1.append(y1)
            all_y2.append(y2)
        all_x1 = torch.stack(all_x1)
        all_x2 = torch.stack(all_x2)

        all_y1 = torch.stack(all_y1).to(device)
        all_y2 = torch.stack(all_y2).to(device)
        x_aug = self.combine_fn(all_x1, all_x2, all_y1, all_y2)
        y_aug = torch.stack((all_y1, all_y2), dim=-1)

        if self.normalized_features:
            x_aug = F.normalize(x_aug, dim=-1)
        return x_aug, y_aug


def str2bool(s):
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def get_noise(x, desired_SNR):
    x_std = x.std()
    # SNR = 10 * log10 ( (signal_power) / (noise_power) )
    # where signal_power = data_std**2 and noise_power = noise_std**2,
    # and SNR is passed as argparse param
    noise_std = x_std / (10 ** (desired_SNR / 20))
    return torch.randn_like(x) * noise_std


class LearnedEmbedding(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("LearnedEmbedding")
        parser.add_argument("--encoder_arch", choices=["basic", "conformer", "vit", "identity"], default="basic")
        parser.add_argument("--clf_arch", choices=["small", "large"], default="small")
        parser.add_argument("--feature_dim", type=int, default=64)
        # Note that with normalized features, we might need to re-normalized after making combinations
        parser.add_argument("--data_noise_SNR", type=float, default=None, help="Desired SNR in dB. None for no noise.")
        parser.add_argument(
            "--feature_noise_SNR", type=float, default=None, help="Desired SNR in dB. None for no noise."
        )
        parser.add_argument("--normalized_features", type=str2bool, default=False)
        parser.add_argument("--feature_combine_type", choices=["avg", "mlp"], default="avg")
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_decay", type=float, default=1.0)
        parser.add_argument("--linearity_loss_coeff", type=float, default=1.0)
        parser.add_argument("--real_CE_loss_coeff", type=float, default=1.0)
        parser.add_argument("--fake_CE_loss_coeff", type=float, default=1.0)
        parser.add_argument("--loss_type", choices=["triplet", "triplet-centroids", "triplet-hard"], default="triplet")
        parser.add_argument("--margin", type=float, default=1.0)
        parser.add_argument("--centroids_momentum", type=float, default=0.75, help="For `triplet-centroids` loss")
        parser.add_argument("--triplets_per_item", type=int, default=1, help="For `triplet` loss")

        parser = parent_parser.add_argument_group("LearnedEmbedding - Fine-tuning")
        parser.add_argument("--finetune_steps", type=int, default=10_000)
        parser.add_argument("--finetune_lr", type=float, default=3e-5)
        parser.add_argument("--finetune_lr_decay", type=float, default=1.0)
        parser.add_argument("--finetune_batch_size", type=float, default=32)
        parser.add_argument("--finetune_test_frac", type=float, default=0.2)
        parser.add_argument("--finetune_n_aug_per_class", type=int, default=-1, help="-1 for all, positive for N")
        return parent_parser

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # Access arg from command line "--arg1" at "self.hparams.arg1", etc

        # NOTE - self.example_input_array - magic pytorch lightning variable for tboard log_graph
        self.example_input_array = torch.ones(1, self.hparams.input_channels, self.hparams.input_time_length)
        if self.hparams.encoder_arch == "basic":
            self.embedding = EmbeddingNetwork(
                input_channels=self.hparams.input_channels,
                input_time_length=self.hparams.input_time_length,
                feature_dim=self.hparams.feature_dim,
                normalized_features=self.hparams.normalized_features,
                use_preprocessed_data=self.hparams.use_preprocessed_data,
            )
        elif self.hparams.encoder_arch == "conformer":
            self.embedding = Conformer(
                feature_dim=self.hparams.feature_dim,
                normalized_features=self.hparams.normalized_features,
            )
        elif self.hparams.encoder_arch == "vit":
            vit = SimpleViT(
                seq_len=962,
                channels=8,
                patch_size=37,
                num_classes=self.hparams.feature_dim,
                dim=256,
                depth=6,
                heads=8,
                mlp_dim=256,
            )
            if self.hparams.normalized_features:
                vit = nn.Sequential(vit, UnitNormLayer())
            self.embedding = vit
        elif self.hparams.arch == "identity":
            self.embedding = DummyIdentity()
        else:
            raise NotImplementedError()
        if self.hparams.clf_arch == "small":
            self.direction_clf = nn.Linear(self.hparams.feature_dim, 5)
            self.modifier_clf = nn.Linear(self.hparams.feature_dim, 5)
        elif self.hparams.clf_arch == "large":
            self.direction_clf = MLPClf(self.hparams.feature_dim, 5)
            self.modifier_clf = MLPClf(self.hparams.feature_dim, 5)
        if self.hparams.loss_type == "triplet":
            self.linearity_loss_fn = TripletLoss(
                margin=self.hparams.margin,
                triplets_per_item=self.hparams.triplets_per_item,
            )
        elif self.hparams.loss_type == "triplet-centroids":
            self.linearity_loss_fn = TripletCentroids(
                margin=self.hparams.margin,
                feature_dim=self.hparams.feature_dim,
                device="cuda" if self.hparams.accelerator == "gpu" else "cpu",
                momentum=self.hparams.centroids_momentum,
            )
        elif self.hparams.loss_type == "triplet-hard":
            self.linearity_loss_fn = TripletLossHardMining(
                margin=self.hparams.margin,
            )
        else:
            logger.error(f"Unknown loss type: {self.hparams.loss_type}")
            raise NotImplementedError()
        if self.hparams.feature_combine_type == "avg":
            # Store on self so it will be detected as additional params
            combine_fn = Avg()
        elif self.hparams.feature_combine_type == "mlp":
            combine_fn = MLPCombine(feature_dim=self.hparams.feature_dim)
        self.feature_combination = CombinePairs(
            combine_fn=combine_fn, normalized_features=self.hparams.normalized_features
        )

    def forward(self, preprocessed_emg_data):
        features = self.embedding(preprocessed_emg_data)
        return features

    def training_step(self, batch, batch_idx):
        (data, labels, is_single, subj_ids) = batch
        # Add noise to each class separately to reach the desired SNR
        if self.hparams.data_noise_SNR is not None:
            with torch.no_grad():
                for label in labels.unique(dim=0):
                    subset_idx = (labels == label).all(-1)
                    subset = data[subset_idx]
                    data[subset_idx] = subset + get_noise(subset, self.hparams.data_noise_SNR)

        # Compute features for real data
        real_features = self.embedding(data)
        # Add noise to features
        if self.hparams.feature_noise_SNR is not None:
            for label in labels.unique(dim=0):
                subset_idx = (labels == label).all(-1)
                subset = real_features[subset_idx]
                real_features[subset_idx] = subset + get_noise(subset, self.hparams.feature_noise_SNR)

        # Create fake double features features from real singles
        single_features = real_features[is_single]
        single_labels = labels[is_single]
        try:
            fake_double_features, fake_double_labels = self.feature_combination(single_features, single_labels)
        except InsufficientDataError:
            logger.warning("Insufficient data for augmentation. Skipping batch.")
            return None

        # Isolate real double features from batch
        real_double_features, real_double_labels = real_features[~is_single], labels[~is_single]
        if len(real_double_features) == 0:
            logger.warning("No real double features in batch. Skipping batch.")
            return None
        if len(fake_double_features) == 0:
            logger.warning("No fake double features in batch. Skipping batch.")
            return None

        # Compute linearity loss
        linearity_loss = self.linearity_loss_fn(
            real_double_features=real_double_features,
            real_double_labels=real_double_labels,
            fake_double_features=fake_double_features,
            fake_double_labels=fake_double_labels,
        )

        # Compute classification loss on real data
        real_dir_logits = self.direction_clf(real_features)
        CE_real_dir = F.cross_entropy(real_dir_logits, labels[:, 0])
        bal_acc_real_dir = accuracy(
            real_dir_logits.argmax(-1), labels[:, 0], task="multiclass", num_classes=5, average="macro"
        )

        real_mod_logits = self.modifier_clf(real_features)
        CE_real_mod = F.cross_entropy(real_mod_logits, labels[:, 1])
        bal_acc_real_mod = accuracy(
            real_mod_logits.argmax(-1), labels[:, 1], task="multiclass", num_classes=5, average="macro"
        )

        # Compute classification loss on fake combinations
        fake_dir_logits = self.direction_clf(fake_double_features)
        CE_fake_dir = F.cross_entropy(fake_dir_logits, fake_double_labels[:, 0])
        bal_acc_fake_dir = accuracy(
            fake_dir_logits.argmax(-1), fake_double_labels[:, 0], task="multiclass", num_classes=5, average="macro"
        )

        fake_mod_logits = self.modifier_clf(fake_double_features)
        CE_fake_mod = F.cross_entropy(fake_mod_logits, fake_double_labels[:, 1])
        bal_acc_fake_mod = accuracy(
            fake_mod_logits.argmax(-1), fake_double_labels[:, 1], task="multiclass", num_classes=5, average="macro"
        )

        # Decrease emphasis on fake CE so they have equal importance
        down_ratio = len(real_features) / len(fake_double_features)
        real_CE = self.hparams.real_CE_loss_coeff * (CE_real_dir + CE_real_mod) / 2
        fake_CE = down_ratio * self.hparams.fake_CE_loss_coeff * (CE_fake_dir + CE_fake_mod) / 2
        lin_loss = self.hparams.linearity_loss_coeff * linearity_loss
        total_loss = real_CE + fake_CE + lin_loss

        # Log individual loss terms (before applying coefficients)
        self.log("train/CE_real_dir", CE_real_dir)
        self.log("train/CE_real_mod", CE_real_mod)
        self.log("train/CE_fake_dir", CE_fake_dir)
        self.log("train/CE_fake_mod", CE_fake_mod)
        self.log("train/linearity_loss", linearity_loss)
        tb = self.logger.experiment
        tb.add_histogram("train/real_double_feature_norm", real_double_features.norm(dim=-1), self.global_step)
        tb.add_histogram("train/fake_double_feature_norm", fake_double_features.norm(dim=-1), self.global_step)

        # Log total loss
        self.log("train/total_loss", total_loss)

        # Log balanced accuracies
        self.log("train/bal_acc_real_dir", bal_acc_real_dir)
        self.log("train/bal_acc_real_mod", bal_acc_real_mod)
        self.log("train/bal_acc_fake_dir", bal_acc_fake_dir)
        self.log("train/bal_acc_fake_mod", bal_acc_fake_mod)
        return total_loss

    def training_epoch_end(self, outputs):
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.callback_metrics.items()}
        metrics = {f"{k}": f"{v:.4f}" for k, v in metrics.items()}
        logger.info(f"Epoch: {self.current_epoch}, Metrics: {metrics}")

    def _val_or_test_step(self, batch, name=None):
        (data, labels, is_single, subj_ids) = batch

        # Compute metrics on real data
        real_features = self.embedding(data)
        real_dir_logits = self.direction_clf(real_features)
        real_mod_logits = self.modifier_clf(real_features)
        real_preds = torch.stack((real_dir_logits.argmax(-1), real_mod_logits.argmax(-1)), dim=-1)
        real_cm = get_combo_conf_mat(labels, real_preds)

        # To be clear that fake data is not part of the result, compute result before making fake data
        res = {"features": real_features, "labels": labels, "is_single": is_single, "subj_ids": subj_ids}

        # Compute metrics on fake data
        single_features = real_features[is_single]
        single_labels = labels[is_single]
        try:
            fake_double_features, fake_double_labels = self.feature_combination(single_features, single_labels)
        except InsufficientDataError:
            logger.warning("Insufficient data for augmentation. Skipping batch.")
            return None
        fake_dir_logits = self.direction_clf(fake_double_features)
        fake_mod_logits = self.modifier_clf(fake_double_features)

        fake_preds = torch.stack((fake_dir_logits.argmax(-1), fake_mod_logits.argmax(-1)), dim=-1)
        fake_cm = get_combo_conf_mat(fake_double_labels, fake_preds)
        if name is not None:
            self.log(f"{name}/single_bal_acc", np.nanmean(np.diag(real_cm)[:8]))
            self.log(f"{name}/double_bal_acc", np.nanmean(np.diag(real_cm)[8:]))
            self.log(f"{name}/overall_bal_acc", np.nanmean(np.diag(real_cm)[:24]))
            self.log(f"{name}/fake_double_bal_acc", np.nanmean(np.diag(fake_cm)[8:]))
        return res

    def validation_step(self, batch, batch_idx):
        self._val_or_test_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._val_or_test_step(batch, None)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_epoch_end(self, outputs):
        features = torch.cat([x["features"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        is_single = torch.cat([x["is_single"] for x in outputs])
        subj_ids = torch.cat([x["subj_ids"] for x in outputs])

        combined_evaluation = self.run_finetune_evaluation(features, labels, is_single, subj_ids)
        scalars = ["single_bal_acc", "double_bal_acc", "overall_bal_acc"]
        for scenario in ["zero_shot", "upper_bound", "lower_bound", "augmented"]:
            for key in scalars:
                value = combined_evaluation[scenario][key]
                self.log(f"test_{scenario}/{key}", value, sync_dist=True)

            # Save confusion matrix
            path = Path(self.logger.log_dir)
            np.save(path / f"test.{scenario}.confusion_matrix.npy", combined_evaluation[scenario]["confusion_matrix"])
        # TODO - how else can we get output from pytorch lightning's trainer.test()?
        return None

    def run_finetune_evaluation(self, features, labels, is_single, subj_ids):
        logger.info("Try evaluation by fine-tuning pre-trained dir_clf and mod_clf")
        # Freeze the feature combination fn, just to be safe
        for param in self.feature_combination.parameters():
            param.requires_grad = False
        self.feature_combination.eval()

        evaluations = []
        for subj_id in subj_ids.unique():
            logger.info(f"Fine-tuning evaluation for subject {subj_id}")
            # Get subset of features and labels for this subject
            idx = subj_ids == subj_id
            evaluations.append(
                self.run_finetune_one_subj(features=features[idx], labels=labels[idx], is_single=is_single[idx])
            )

        combined_evaluation = {}
        for key in ["upper_bound", "lower_bound", "augmented", "zero_shot"]:
            combined_evaluation[key] = {
                "single_bal_acc": np.mean([x[key]["single_bal_acc"] for x in evaluations]),
                "double_bal_acc": np.mean([x[key]["double_bal_acc"] for x in evaluations]),
                "overall_bal_acc": np.mean([x[key]["overall_bal_acc"] for x in evaluations]),
                "confusion_matrix": np.mean([x[key]["confusion_matrix"] for x in evaluations], axis=0),
            }
        return combined_evaluation

    def run_finetune_one_subj(self, features, labels, is_single):
        # Split into train/test
        N_single = is_single.sum().item()
        N_single_test = int(N_single * self.hparams.finetune_test_frac)

        N_double = (~is_single).sum().item()
        N_double_test = int(N_double * self.hparams.finetune_test_frac)

        np.random.seed(0)
        single_perm = np.random.permutation(N_single)
        test_single_feat = features[is_single][single_perm[:N_single_test]]
        test_single_labels = labels[is_single][single_perm[:N_single_test]]
        train_single_feat = features[is_single][single_perm[N_single_test:]]
        train_single_labels = labels[is_single][single_perm[N_single_test:]]

        double_perm = np.random.permutation(N_double)
        test_double_feat = features[~is_single][double_perm[:N_double_test]]
        test_double_labels = labels[~is_single][double_perm[:N_double_test]]
        train_double_feat = features[~is_single][double_perm[N_double_test:]]
        train_double_labels = labels[~is_single][double_perm[N_double_test:]]

        def try_once(which: str):
            logger.info(f"Finetune for scenario: {which}")
            aug = {"upper": None, "lower": None, "aug": self.feature_combination}[which]
            doubles_in_train = {"upper": True, "lower": False, "aug": False}[which]

            # Setup train data
            logger.debug(f"real singles: {len(train_single_feat)}")
            logger.debug(f"real doubles: {len(train_double_feat)}")
            if doubles_in_train:
                x_train = torch.cat((train_single_feat, train_double_feat))
                y_train = torch.cat((train_single_labels, train_double_labels))
            else:
                x_train = train_single_feat
                y_train = train_single_labels
            if aug is not None:
                x_aug, y_aug = aug(train_single_feat, train_single_labels)
                if self.hparams.finetune_n_aug_per_class > 0:
                    # Subset each class
                    res_x, res_y = [], []
                    for c in y_aug.unique(dim=0):
                        idx = (y_aug == c).all(dim=1)
                        perm = np.random.permutation(idx.sum().item())
                        res_x.append(x_aug[idx][perm[: self.hparams.finetune_n_aug_per_class]])
                        res_y.append(y_aug[idx][perm[: self.hparams.finetune_n_aug_per_class]])
                    x_aug, y_aug = torch.cat(res_x), torch.cat(res_y)
                logger.debug(f"n_aug_per_class: {self.hparams.finetune_n_aug_per_class}")
                logger.debug(f"fake doubles: {x_aug.shape[0]}")
                x_train = torch.cat([x_train, x_aug])
                y_train = torch.cat([y_train, y_aug])

            x_train, y_train = shuffle_together(x_train, y_train)

            # Setup test data
            x_test = torch.cat([test_single_feat, test_double_feat])
            y_test = torch.cat([test_single_labels, test_double_labels])
            x_test, y_test = shuffle_together(x_test, y_test)

            # Make a temporary copy of the models
            dir_clf = deepcopy(self.direction_clf)
            mod_clf = deepcopy(self.modifier_clf)
            optim = torch.optim.AdamW(chain(dir_clf.parameters(), mod_clf.parameters()), lr=self.hparams.finetune_lr)
            sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.hparams.finetune_lr_decay)
            # Since the features are already on GPU, can't use multiprocess dataloader
            bs = self.hparams.finetune_batch_size
            train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True, num_workers=0)
            test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=bs, shuffle=False, num_workers=0)

            def infinite_cycle(loader):
                while True:
                    for x, y in loader:
                        yield x, y

            inf_train_loader = infinite_cycle(train_loader)

            @torch.no_grad()
            def test():
                dir_clf.eval()
                mod_clf.eval()
                dir_logits, mod_logits, y_test = [], [], []
                for x, y in test_loader:
                    dir_logits.append(dir_clf(x))
                    mod_logits.append(mod_clf(x))
                    y_test.append(y)
                dir_logits = torch.cat(dir_logits)
                mod_logits = torch.cat(mod_logits)
                y_test = torch.cat(y_test)
                preds = torch.stack((dir_logits.argmax(-1), mod_logits.argmax(-1)), dim=-1)
                cm = get_combo_conf_mat(y_test, preds)
                return {
                    "single_bal_acc": np.nanmean(np.diag(cm)[:8]),
                    "double_bal_acc": np.nanmean(np.diag(cm)[8:]),
                    "overall_bal_acc": np.nanmean(np.diag(cm)[:24]),
                    "confusion_matrix": cm,
                }

            zero_shot_res = test()  # Test once with no fine-tuning
            logger.debug(f"Zero-shot results: {zero_shot_res}")
            tb = self.logger.experiment
            # Graphs will start with the zero-shot result
            scalars = ["single_bal_acc", "double_bal_acc", "overall_bal_acc"]
            for k in scalars:
                v = zero_shot_res[k]
                tb.add_scalar(f"finetune/{which}/{k}", v, 0)  # Start at x-axis=0
            # Continue graphs from 1 onward
            dir_clf.train()
            mod_clf.train()
            for i in range(1, self.hparams.finetune_steps + 1):
                x, y = next(inf_train_loader)
                optim.zero_grad()
                dir_logits = dir_clf(x)
                mod_logits = mod_clf(x)
                dir_loss = F.cross_entropy(dir_logits, y[:, 0])
                mod_loss = F.cross_entropy(mod_logits, y[:, 1])
                loss = dir_loss + mod_loss
                loss.backward()
                optim.step()

                if i % 100 == 0:
                    finetuned_res = test()
                    logger.debug(f"Step {i} results: {finetuned_res}")
                    for k in scalars:
                        v = finetuned_res[k]
                        tb.add_scalar(f"finetune/{which}/{k}", v, i)
                    dir_clf.train()
                    mod_clf.train()
                    sched.step()

            finetuned_res = test()

            return finetuned_res, zero_shot_res

        upper, zero_shot_res = try_once("upper")
        lower, _ = try_once("lower")
        aug, _ = try_once("aug")
        return {"zero_shot": zero_shot_res, "upper_bound": upper, "lower_bound": lower, "augmented": aug}

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.embedding.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.hparams.lr_decay)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "name": "lr_sched"}}


def try_finetune_from_ckpt(ckpt: Path, **kwargs):
    # Convenience function - for trying the fine-tuning stage on a pre-trained model
    pl.seed_everything(0)
    hparams = torch.load(ckpt)["hyper_parameters"]
    hparams.update(kwargs)
    per_subj_data = get_per_subj_data()
    datamodule = DataModule(per_subj_data=per_subj_data, **hparams)

    learned_embedding = LearnedEmbedding.load_from_checkpoint(ckpt, **kwargs)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(PROJECT_PATH / "results_dev", name=hparams["name"], default_hp_metric=False),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        deterministic=True,
    )
    trainer.test(learned_embedding, datamodule=datamodule)
