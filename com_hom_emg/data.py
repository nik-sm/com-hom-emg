import pickle
import random
from typing import Tuple

import numpy as np
import torch
from loguru import logger
from pytorch_lightning import LightningDataModule
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, TensorDataset

from .utils import DIRECTION_GESTURES, MODIFIER_GESTURES, PROJECT_PATH


def get_canonical_coords():
    """Specifies how labels will appear on our confusion matrices,
    so that we can convert from 2d_label to position_in_conf_mat"""
    canonical_coords = []
    canonical_coords_str = []
    for i, d in enumerate(DIRECTION_GESTURES):
        canonical_coords.append((i, 4))
        canonical_coords_str.append(f"({d}, None)")
    for i, m in enumerate(MODIFIER_GESTURES):
        canonical_coords.append((4, i))
        canonical_coords_str.append(f"(None, {m})")
    for i, d in enumerate(DIRECTION_GESTURES):
        for j, m in enumerate(MODIFIER_GESTURES):
            canonical_coords.append((i, j))
            canonical_coords_str.append(f"({d}, {m})")
    canonical_coords.append((4, 4))
    canonical_coords_str.append("(None, None)")
    return canonical_coords, canonical_coords_str


def get_per_subj_data():
    path = PROJECT_PATH / "data" / "self-contained-data.combination-gestures.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def onehot(y_integer: np.ndarray):
    """
    Args:
        y_integer (np.ndarray): shape (items, 2) - integer label for direction and modifier

    Returns:
        y_onehot (np.ndarray): shape (items, 2, classes) - one-hot labels for direction and modifier
    """
    y_onehot = np.zeros((y_integer.shape[0], 2, 5))
    y_onehot[np.arange(y_integer.shape[0]), 0, y_integer[:, 0]] = 1
    y_onehot[np.arange(y_integer.shape[0]), 1, y_integer[:, 1]] = 1
    return y_onehot


def str2bool(s):
    # TODO - duplicated
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def shuffle_together(*tensors):
    """Shuffle tensors together"""
    assert all(isinstance(x, torch.Tensor) for x in tensors)
    assert all(len(x) == len(tensors[0]) for x in tensors)
    p = torch.randperm(len(tensors[0]))
    return [x[p] for x in tensors]


class DataModule(LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--fold", type=int, required=True)
        parser.add_argument("--n_train_subj", type=int, default=8)
        parser.add_argument("--n_val_subj", type=int, default=1)
        parser.add_argument("--n_test_subj", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--use_preprocessed_data", type=str2bool, default=False)
        return parent_parser

    def __init__(
        self,
        *,
        # seed and per_subj_data come from cli
        seed: int,
        per_subj_data: dict,
        #
        fold: int,
        n_train_subj: int,
        n_val_subj: int,
        n_test_subj: int,
        batch_size: int,
        num_workers: int,
        use_preprocessed_data: bool,
        **kw,
    ):
        """
        From N subjects, we select 1 for val, 1 for test, and N-2 for train.
        In each set, data are merged and shuffled.
        While loading, we distinguish single and double gestures for easier splitting during train steps.
        """
        super().__init__()
        self.train_set, self.val_set, self.test_set = get_datasets(
            per_subj_data, fold, n_train_subj, n_val_subj, n_test_subj, use_preprocessed_data
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.example_data_shape = self.train_set.tensors[0][0].shape

    def get_loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.get_loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self.get_loader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self.get_loader(self.test_set, shuffle=False)


def get_datasets(
    per_subj_data: dict,
    fold: int,
    n_train_subj: int,
    n_val_subj: int,
    n_test_subj: int,
    use_preprocessed_data: bool,
    return_subj_names: bool = False,  # For testing
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Separate subjects; some for learning embedding, some for val and some for test.
    (Unseen subjects for val and test)
    Returns train, val, test datasets.
    """
    assert fold in range(len(per_subj_data))

    def collect_one(_subjs, subj_id_offset=0):
        data, labels, is_single, subj_ids = [], [], [], []
        for subj_id, subj in enumerate(_subjs):
            # Add doubles
            x = per_subj_data[subj]["doubles"]["data"]
            data.append(x)
            labels.append(per_subj_data[subj]["doubles"]["2d_labels"])
            is_single.append(np.zeros(len(x), dtype=bool))  # NOTE - careful to use bool dtype; used for masking later
            subj_ids.append((subj_id + subj_id_offset) * np.ones(len(x), dtype=int))

            # Add singles
            for key in ["calibration", "held_singles", "pulsed_singles"]:
                x = per_subj_data[subj][key]["data"]
                data.append(x)
                labels.append(per_subj_data[subj][key]["2d_labels"])
                is_single.append(np.ones(len(x), dtype=bool))
                subj_ids.append((subj_id + subj_id_offset) * np.ones(len(x), dtype=int))

        data = np.concatenate(data)
        if use_preprocessed_data:
            data = preprocess(data)
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(np.concatenate(labels))
        is_single = torch.from_numpy(np.concatenate(is_single))
        subj_ids = torch.from_numpy(np.concatenate(subj_ids))
        data, labels, is_single, subj_ids = shuffle_together(data, labels, is_single, subj_ids)
        return TensorDataset(data, labels, is_single, subj_ids)

    subjs = np.roll(list(per_subj_data.keys()), -fold)
    if n_train_subj + n_val_subj + n_test_subj != len(subjs):
        raise ValueError(f"Num subjects in train/val/test splits must sum to {len(subjs)}")

    test_subj = subjs[0:n_test_subj]
    val_subj = subjs[n_test_subj : n_test_subj + n_val_subj]
    train_subj = subjs[n_test_subj + n_val_subj :]

    assert np.intersect1d(train_subj, val_subj).size == 0
    assert np.intersect1d(train_subj, test_subj).size == 0
    assert np.intersect1d(val_subj, test_subj).size == 0

    train_set, val_set, test_set = collect_one(train_subj), collect_one(val_subj), collect_one(test_subj)
    logger.info(f"Train subjects: {train_subj}")
    logger.info(f"Val subjects: {val_subj}")
    logger.info(f"Test subjects: {test_subj}")
    logger.info(f"Train on {len(train_subj)} subjects:\n{[x.shape for x in train_set.tensors]}")
    logger.info(f"Validate on {len(val_subj)} subjects:\n{[x.shape for x in val_set.tensors]}")
    logger.info(f"Test on {len(test_subj)} subjects:\n{[x.shape for x in test_set.tensors]}")
    if not return_subj_names:
        return train_set, val_set, test_set
    return train_set, val_set, test_set, train_subj, val_subj, test_subj


def bandpass(data, lo, hi, fs, order=5):
    sos = butter(order, [lo, hi], fs=fs, analog=False, btype="band", output="sos")
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


def preprocess(data: np.ndarray) -> np.ndarray:
    data = bandpass(data, lo=5, hi=450, fs=1926)
    # data = np.abs(data)
    data = data[..., ::4]  # NOTE - 1926 / 4 = 481.5, greater than our bandpass upper limit
    data -= data.mean()
    data /= data.std()
    data = np.ascontiguousarray(data)
    # TODO - after tuning everything else, compare raw data vs preprocessing.
    # Need to also remove layers when downsampling data.
    # example pipeline idea: bandpass, rectify, downsample, rescale, smooth
    # logger.critical("TODO - after tuning other factors, decide about data preprocessing")
    return data
