from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator, KFold

from .data import onehot
from .utils import DIRECTION_GESTURES, MODIFIER_GESTURES

# Make a lookup table so we can convert from 2d_label to position_in_conf_mat
CANONICAL_COORDS = []
CANONICAL_COORDS_STR = []
for i, d in enumerate(DIRECTION_GESTURES):
    CANONICAL_COORDS.append((i, 4))
    CANONICAL_COORDS_STR.append(f"({d}, None)")
for i, m in enumerate(MODIFIER_GESTURES):
    CANONICAL_COORDS.append((4, i))
    CANONICAL_COORDS_STR.append(f"(None, {m})")
for i, d in enumerate(DIRECTION_GESTURES):
    for j, m in enumerate(MODIFIER_GESTURES):
        CANONICAL_COORDS.append((i, j))
        CANONICAL_COORDS_STR.append(f"({d}, {m})")
CANONICAL_COORDS.append((4, 4))
CANONICAL_COORDS_STR.append("(None, None)")


def get_combo_conf_mat(y_true_2d, y_pred_2d, normalize=True):
    """We get a confusion matrix of shape (25, 25). Row is true class, col is predicted.
    Entries are arranged like this:
        (D1, None), ..., (D4, None), (None, M1), ..., (None, M4), (D1, M1), ...
        (D1, M4), (D2, M1), ... (D2, M4), ... (D4, M4), (None, None)
    where D1 ... D4 are directions in order of appearance from DIRECTION_GESTURES
    and M1 ... M4 are modifiers in order of appearance from MODIFIER_GESTURES.
    This means the first 4 rows are each "direction-only" label, next 4 are "modifier-only" labels."""
    cm = np.zeros((len(CANONICAL_COORDS), len(CANONICAL_COORDS)))
    for yt, yp in zip(y_true_2d, y_pred_2d):
        cm[CANONICAL_COORDS.index(tuple(yt)), CANONICAL_COORDS.index(tuple(yp))] += 1
    if normalize:
        # NOTE - result may contain nans - use nanmean later
        with np.errstate(all="ignore"):  # Ignore division by zero for empty rows
            cm /= cm.sum(axis=-1, keepdims=True)
    return cm


class CVIterableWrapper(BaseCrossValidator):
    def __init__(self, cv_iterable):
        self.cv_iterable = cv_iterable

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.cv_iterable)

    def split(self, X=None, y=None, groups=None):
        for train, test in self.cv_iterable:
            yield train, test


@dataclass
class ClassCounts:
    train_classes: np.ndarray
    train_counts: np.ndarray
    test_classes: np.ndarray
    test_counts: np.ndarray
    n_aug: int = 0


@torch.no_grad()
def run_one_fold_augmented(model, x, y, train_idx, test_idx, aug_fn: Optional[Callable], dry_run: bool):
    assert len(np.intersect1d(train_idx.flatten(), test_idx.flatten())) == 0, "Train and test overlap"

    # Get class counts before adding augmented data and before converting to onehot
    train_classes, train_counts = np.unique(y[train_idx], axis=0, return_counts=True)
    test_classes, test_counts = np.unique(y[test_idx], axis=0, return_counts=True)

    if aug_fn is not None:
        x_aug, y_aug = aug_fn(x[train_idx], y[train_idx])
        if isinstance(x_aug, torch.Tensor):
            x_aug = x_aug.cpu().numpy()
            y_aug = y_aug.cpu().numpy()

        n_aug = len(x_aug)
        x_train = np.concatenate([x[train_idx], x_aug], axis=0)
        y_real = onehot(y[train_idx]) if aug_fn.makes_3d_labels else y[train_idx]
        y_train = np.concatenate((y_real, y_aug), axis=0)
    else:
        n_aug = 0
        x_train = x[train_idx]
        y_train = y[train_idx]

    if dry_run:
        x_train, y_train = x_train[:50], y_train[:50]
    model.fit(x_train, y_train)
    y_true = np.copy(y[test_idx])
    y_pred = model.predict(x[test_idx])

    return y_true, y_pred, ClassCounts(train_classes, train_counts, test_classes, test_counts, n_aug)


def get_cv_conf_mats_with_augmentation(
    model, x: np.ndarray, y: np.ndarray, cv: Union[int, Iterable], aug_fn: Optional[Callable], n_jobs=-1, dry_run=False
):
    """
    Like sklearn.model_selection.cross_val_predict, but returns predictions from each fold separately.
    Allows us to compute avg and std confusion matrix across folds, giving info on variability.
    """
    if isinstance(cv, int):
        kf = KFold(cv)
    else:
        kf = CVIterableWrapper(cv)

    model = clone(model)

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_fold_augmented)(model, x, y, train_idx, test_idx, aug_fn, dry_run)
        for train_idx, test_idx in kf.split(x, y)
    )
    cv_labels, cv_preds, cv_class_counts = list(zip(*results))
    conf_mats = [get_combo_conf_mat(yt, yp) for yt, yp in zip(cv_labels, cv_preds)]
    return conf_mats, cv_class_counts
