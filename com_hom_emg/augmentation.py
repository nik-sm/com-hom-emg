from abc import ABC, abstractmethod
from itertools import product
from typing import Tuple, Union

import numpy as np
import torch

from .data import onehot

ArrayDef = Union[np.ndarray, torch.Tensor]


def unique(_y):
    return torch.unique(_y, dim=0) if isinstance(_y, torch.Tensor) else np.unique(_y, axis=0)


def concat(_x):
    return torch.cat(_x, dim=0) if isinstance(_x[0], torch.Tensor) else np.concatenate(_x, axis=0)


def stack(_x):
    return torch.stack(_x, dim=0) if isinstance(_x[0], torch.Tensor) else np.stack(_x, axis=0)


def eq(x, y):
    return torch.eq(x, y) if isinstance(x, torch.Tensor) else np.equal(x, y)


def where(x):
    return torch.where(x)[0] if isinstance(x, torch.Tensor) else np.where(x)[0]


class InsufficientDataError(Exception):
    ...


def split_dir_mod(x: ArrayDef, y: ArrayDef, n_per_class: int):
    """Given single and double data:
    - isolate the singles
    - split into direction and modifier gestures
    - split these by class

    Returns:
        x_dir List[Union[torch.tensor, np.ndarray]]: each item shape (n_samples, n_features)
        x_mod List[Union[torch.tensor, np.ndarray]]: each item shape (n_samples, n_features)
        y_dir List[Union[torch.tensor, np.ndarray]]: each item shape (n_samples) - integer labels
        y_mod List[Union[torch.tensor, np.ndarray]]: each item shape (n_samples) - integer labels
    """
    singles_idx = where((y[:, 0] == 4) | (y[:, 1] == 4))
    x_single, y_single = x[singles_idx], y[singles_idx]

    # Separate into direction gestures and modifier gestures
    y_dir_all = y_single[y_single[:, 0] != 4]
    x_dir_all = x_single[y_single[:, 0] != 4]
    y_mod_all = y_single[y_single[:, 1] != 4]
    x_mod_all = x_single[y_single[:, 1] != 4]

    # Sanity check that we have single gestures
    assert all(y_dir_all[:, 1] == 4), "Modifier should be 'none' for all direction items"
    assert all(y_mod_all[:, 0] == 4), "Direction should be 'none' for all modifier items"
    if len(y_dir_all) == 0:
        raise InsufficientDataError
    if len(y_mod_all) == 0:
        raise InsufficientDataError

    x_dir, y_dir = [], []
    for y in unique(y_dir_all):
        x_this_class = x_dir_all[eq(y_dir_all, y).all(-1)]
        y_this_class = y_dir_all[eq(y_dir_all, y).all(-1)][:, 0]
        if n_per_class > 0:
            x_this_class = x_this_class[:n_per_class]
            y_this_class = y_this_class[:n_per_class]
        x_dir.append(x_this_class)
        y_dir.append(y_this_class)
    x_dir, y_dir = concat(x_dir), concat(y_dir)

    x_mod, y_mod = [], []
    for y in unique(y_mod_all):
        x_this_class = x_mod_all[eq(y_mod_all, y).all(-1)]
        y_this_class = y_mod_all[eq(y_mod_all, y).all(-1)][:, 1]
        if n_per_class > 0:
            x_this_class = x_this_class[:n_per_class]
            y_this_class = y_this_class[:n_per_class]
        x_mod.append(x_this_class)
        y_mod.append(y_this_class)
    x_mod, y_mod = concat(x_mod), concat(y_mod)

    return x_dir, x_mod, y_dir, y_mod


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, x_single: np.ndarray, y_single: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x_single: (n_samples_in, n_features) - data/features from single gestures
            y_single: (n_samples_in, 2) - 2D labels of data/features from single gestures

        Returns:
            x_prime: (n_samples_aug, n_features) - augmented data
            y_prime: (n_samples_aug, 2) - augmented labels
        """

    @abstractmethod
    def __repr__(self):
        ...

    @property
    @abstractmethod
    def makes_3d_labels(self) -> bool:
        ...


class Stack(Augmentation):
    """Create several chunks of augmented data and stack them together."""

    def __init__(self, *augmentations):
        self.augmentations = augmentations

    def __call__(self, x: ArrayDef, y: ArrayDef, **kwargs) -> Tuple[ArrayDef, ArrayDef]:
        x_prime, y_prime = [], []
        for aug in self.augmentations:
            x_aug, y_aug = aug(x, y, **kwargs)
            x_prime.append(x_aug)
            y_prime.append(y_aug)
        if any([aug.makes_3d_labels for aug in self.augmentations]):
            y_prime = [onehot(y) if not aug.makes_3d_labels else y for aug, y in zip(self.augmentations, y_prime)]
        return concat(x_prime), concat(y_prime)

    def __repr__(self):
        return f"{type(self).__name__}({', '.join([repr(aug) for aug in self.augmentations])})"

    @property
    def makes_3d_labels(self) -> bool:
        return any([aug.makes_3d_labels for aug in self.augmentations])


class AvgPairs(Augmentation):
    """Create fake doubles by averaging pairs of singles. New items have hard labels including both classes"""

    makes_3d_labels = False

    def __init__(self, n_per_class: int):
        self.n_per_class = n_per_class

    def __call__(self, x: ArrayDef, y: ArrayDef):
        x_dir, x_mod, y_dir, y_mod = split_dir_mod(x, y, self.n_per_class)
        x_aug, y_aug = [], []
        for (x1, y1), (x2, y2) in product(zip(x_dir, y_dir), zip(x_mod, y_mod)):
            x_aug.append((x1 + x2) / 2)
            if isinstance(x, torch.Tensor):
                label = torch.tensor([y1, y2], device=x.device)
            else:
                label = np.array([y1, y2])
            y_aug.append(label)
        return stack(x_aug), stack(y_aug)

    def __repr__(self):
        return f"{type(self).__name__}(n_per_class={self.n_per_class})"


class SoftAvgPairs(Augmentation):
    """Create fake doubles by averaging pairs of singles,
    but give some probability mass to `NoDirection` and `NoModifier` classes."""

    makes_3d_labels = True

    def __init__(self, weight, n_per_class: int):
        self.weight = weight
        self.n_per_class = n_per_class

    def __call__(self, x: ArrayDef, y: ArrayDef):
        x_dir, x_mod, y_dir, y_mod = split_dir_mod(x, y, self.n_per_class)
        x_aug, y_aug = [], []
        for (x1, y1), (x2, y2) in product(zip(x_dir, y_dir), zip(x_mod, y_mod)):
            x_aug.append((x1 + x2) / 2)
            label = np.zeros((2, 5)) if isinstance(x, np.ndarray) else torch.zeros(2, 5)
            label[0, y1] = self.weight
            label[0, 4] = 1 - self.weight
            label[1, y2] = self.weight
            label[1, 4] = 1 - self.weight
            y_aug.append(label)
        return stack(x_aug), stack(y_aug)

    def __repr__(self):
        return f"{type(self).__name__}(weight={self.weight}, n_per_class={self.n_per_class})"


class MixUpPairs(Augmentation):
    """Create fake doubles by making random convex mixtures."""

    makes_3d_labels = True

    def __init__(self, n_items_per_pair: int, beta_param: float, n_per_class: int):
        """
        Args:
            n_items_per_pair (int, optional):
            beta_param (float, optional): mixture weights sampled from Beta(beta_param, beta_param).
                1.0 => uniform weights in [0, 1]
                <1 => weights concentrated near 0 and near 1
                >1 => weights concentrated near 0.5
            n_per_class (int): number of samples used per class
        """
        self.n_items_per_pair = n_items_per_pair
        self.beta_param = beta_param
        self.n_per_class = n_per_class

    def __call__(self, x: ArrayDef, y: ArrayDef):
        x_dir, x_mod, y_dir, y_mod = split_dir_mod(x, y, self.n_per_class)
        x_aug, y_aug = [], []
        for (x1, y1), (x2, y2) in product(zip(x_dir, y_dir), zip(x_mod, y_mod)):
            for _ in range(self.n_items_per_pair):
                alpha = np.random.beta(self.beta_param, self.beta_param)
                x_aug.append(alpha * x1 + (1 - alpha) * x2)
                label = np.zeros((2, 5)) if isinstance(x, np.ndarray) else torch.zeros(2, 5)
                label[0, y1] = alpha
                label[0, 4] = 1 - alpha  # NOTE - assign remaining mass to "none"
                label[1, y2] = 1 - alpha
                label[1, 4] = alpha  # NOTE - assign remaining mass to "none"
                y_aug.append(label)
        return stack(x_aug), stack(y_aug)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"n_items_per_pair={self.n_items_per_pair}, "
            + f"beta_param={self.beta_param}, n_per_class={self.n_per_class}))"
        )


class WhiteNoise(Augmentation):
    """Add noise on each feature, where noise scale is relative to the feature's standard deviation."""

    makes_3d_labels = False

    def __init__(self, n_per_item: int, relative_sigma: float):
        self.n_per_item = n_per_item
        self.relative_sigma = relative_sigma

    def __call__(self, x_all: np.ndarray, y_all: np.ndarray):
        x_dir, x_mod, y_dir, y_mod = split_dir_mod(x_all, y_all, -1)
        feature_stds = concat((x_dir, x_mod)).std(0) * self.relative_sigma
        x_aug, y_aug = [], []
        for x, y in zip(x_dir, y_dir):
            for _ in range(self.n_per_item):
                noise = np.random.randn(len(x)).astype(np.float32) * feature_stds
                x_aug.append(x + noise)  # (Tensor + ndarray) or (ndarray + ndarray), but not (ndarray + Tensor)
                this_y = np.array([y, 4]) if isinstance(x, np.ndarray) else torch.tensor([y, 4])
                y_aug.append(this_y)
        for x, y in zip(x_mod, y_mod):
            for _ in range(self.n_per_item):
                noise = np.random.randn(len(x)).astype(np.float32) * feature_stds
                x_aug.append(x + noise)
                this_y = np.array([4, y]) if isinstance(x, np.ndarray) else torch.tensor([4, y])
                y_aug.append(this_y)
        return stack(x_aug), stack(y_aug)

    def __repr__(self):
        return f"{type(self).__name__}(n_per_item={self.n_per_item}, relative_sigma={self.relative_sigma})"
