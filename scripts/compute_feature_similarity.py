"""Compute similarity between features of real and fake combos"""
import argparse
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from ablation_settings import settings_names as ablation_settings_names
from loguru import logger
from pytorch_lightning import seed_everything
from regular_settings import settings_names as regular_settings_names
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm

from com_hom_emg.data import DataModule, get_per_subj_data
from com_hom_emg.model import LearnedEmbedding
from com_hom_emg.utils import DIRECTION_GESTURES, MODIFIER_GESTURES, PROJECT_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class FailedRunError(Exception):
    pass


def load_one(folder: Path, which="best"):
    """Extract train, val, and test metrics from the specified checkpoint (best or last).
    Also extract hyperparams from hparams.yaml file."""

    # Given a checkpoint like this: best__epoch=38__step=14664__val_aug_overall_acc=0.569.ckpt
    # We want to extract the step: 14664
    ckpts = folder / "checkpoints"
    matching_ckpts = list(ckpts.glob(f"{which}*.ckpt"))
    if len(matching_ckpts) == 0:
        raise FailedRunError(f"No checkpoint found for {which} in {folder}")
    # When there are multiple runs, take the most recent.
    # Since only 1 metrics.csv is kept, this matches the latest ckpt
    chosen_ckpt = max(matching_ckpts, key=lambda x: x.stat().st_mtime)

    hparams = yaml.safe_load((folder / "hparams.yaml").read_text())

    return hparams, chosen_ckpt


def subset_one_class(X, Y, N):
    idx = np.random.choice(len(X), size=N, replace=False)
    return X[idx], Y[idx]


def subset_each_class(X, Y, N):
    result_x, result_y = [], []
    for y in Y.unique(dim=0):
        idx = (Y == y).all(-1)
        x = X[idx]
        y = Y[idx]
        x, y = subset_one_class(x, y, N)
        result_x.append(x)
        result_y.append(y)
    return torch.cat(result_x), torch.cat(result_y)


def compute_feature_similarity_no_labels(features: List[np.ndarray], gamma: float) -> np.ndarray:
    """Compute NxN matrix of similarity between each pair of classes in feature space

    Returns:
        np.ndarray: similarity between each pair of classes
    """
    if not isinstance(features, list):
        raise ValueError("features must be list of chunks of data")
    # Start with NaN so that missing gestures will not skew towards default value.
    # NOTE - Later, need to use functions like np.nanmean(), etc
    n_class = len(features)
    pairwise_similarity = np.nan * np.ones((n_class, n_class))

    for (idx1, feat1), (idx2, feat2) in combinations(enumerate(features), 2):
        # Match rows based on their full label vector
        rbf_similarities = np.exp(-gamma * cdist(feat1, feat2, "sqeuclidean"))
        pairwise_similarity[idx1, idx2] = pairwise_similarity[idx2, idx1] = rbf_similarities.mean()
    # Fill diagonal.  different because we must ignore zeros due to d(x1, x1) == 0
    for idx, feat in enumerate(features):
        if len(feat) <= 1:
            continue
        rbf_similarities = np.exp(-gamma * pdist(feat, "sqeuclidean"))
        pairwise_similarity[idx, idx] = rbf_similarities.mean()

    return pairwise_similarity


@torch.no_grad()
def compute_similarity_one_ckpt(ckpt, gamma: Optional[float]) -> Tuple[np.ndarray, List[str]]:
    """Compute similarity matrix for a single checkpoint

    Returns
        similarity matrix (shape 32 x 32)
        list of class names (length 32)
    """
    embedding = LearnedEmbedding.load_from_checkpoint(ckpt)
    embedding.eval()

    per_subj_data = get_per_subj_data()
    datamodule = DataModule(per_subj_data=per_subj_data, **embedding.hparams)
    test_loader = datamodule.test_dataloader()

    # Get features of real double gestures
    embedding.to(device)
    features, labels, is_single = [], [], []
    for batch_data, batch_labels, batch_is_single, _subj_ids in test_loader:
        features.append(embedding(batch_data.to(device)))
        labels.append(batch_labels.to(device))
        is_single.append(batch_is_single)
    features = torch.cat(features)
    labels = torch.cat(labels)
    is_single = torch.cat(is_single)

    single_feat = features[is_single]
    single_labels = labels[is_single]
    double_feat = features[~is_single]
    double_labels = labels[~is_single]

    fake_features, fake_labels = embedding.feature_combination(single_feat, single_labels)
    fake_features, fake_labels = subset_each_class(fake_features, fake_labels, N=500)

    # Convert to numpy
    double_feat = double_feat.cpu().numpy()
    double_labels = double_labels.cpu().numpy()
    fake_features = fake_features.cpu().numpy()
    fake_labels = fake_labels.cpu().numpy()

    combo_labels = np.unique(fake_labels, axis=0)
    assert (combo_labels == 4).sum() == 0  # Make sure no single gestures are included

    ticktext = []
    all_features_grouped = []
    # Add a chunk of data for each real combo class
    for label in combo_labels:
        idx = (double_labels == label).all(-1)
        all_features_grouped.append(double_feat[idx])
        d, m = label
        dir_name = DIRECTION_GESTURES[d]
        mod_name = MODIFIER_GESTURES[m]
        ticktext.append(f"({dir_name}, {mod_name}) - Real")

    # Add the fake doubles, 1 class at a time
    for label in combo_labels:
        idx = (fake_labels == label).all(-1)
        all_features_grouped.append(fake_features[idx])
        d, m = label
        dir_name = DIRECTION_GESTURES[d]
        mod_name = MODIFIER_GESTURES[m]
        ticktext.append(f"({dir_name}, {mod_name}) - Fake")

    if gamma is None:
        # Compute lengthscale for RBF kernel using median heuristic
        gamma = 1 / np.median(pdist(np.concatenate(all_features_grouped), "sqeuclidean"))
    similarity_matrix = compute_feature_similarity_no_labels(all_features_grouped, gamma)

    return similarity_matrix, ticktext


def compute_similarities(settings: List[dict], cols: List[str], results_dir: Path, title: str, gamma: Optional[float]):
    settings = deepcopy(settings)
    failed_runs = []
    missing_runs = []

    records = []
    for folder in tqdm(list(results_dir.iterdir()), desc="folders"):
        if len(settings) == 0:
            logger.info("All settings have been found - stopping search")
            break

        if folder.name == "slurm_logs":
            continue

        if folder.is_dir():
            try:
                idx = settings.index(folder.name)
            except ValueError:
                # This run is not part of the settings we are interested in
                logger.info(f"Skipping {folder.name} - not in settings")
                continue
            settings.pop(idx)  # Now that we've found this setting, remove from the list

            # Try to load this setting
            try:
                hparams, ckpt = load_one(folder)
            except FailedRunError:
                failed_runs.append(folder)
                continue
            similarity_matrix, ticktext = compute_similarity_one_ckpt(ckpt, gamma=gamma)

            # Construct a row, starting with hyperparams
            record = {k: hparams[k] for k in cols}
            # record["median_errors"] = median_errors
            record["similarity_matrix"] = similarity_matrix
            record["ticktext"] = ticktext
            records.append(record)

    # At the end, all remaining settings are missing
    missing_runs = settings

    failed_runs_file = PROJECT_PATH / f"FAILED_RUNS.{title}.txt"
    if len(failed_runs) > 0:
        logger.warning(f"Failed to load {len(failed_runs)} runs")
        with open(failed_runs_file, "w") as f:
            for failed_run in failed_runs:
                print(failed_run.name, file=f)

    missing_runs_file = PROJECT_PATH / f"MISSING_RUNS.{title}.txt"
    if len(missing_runs) > 0:
        logger.warning(f"Missing {len(missing_runs)} runs")
        with open(missing_runs_file, "w") as f:
            for missing_run in missing_runs:
                print(missing_run, file=f)

    return pd.DataFrame.from_records(records)


@logger.catch(onerror=lambda _: sys.exit(1))
def main(results_dir: Path, figs_dir: Path, which_expt: str, gamma: Optional[float]):
    seed_everything(0)
    title = f"feature_similarity.{which_expt}.{gamma}"
    if which_expt == "regular":
        settings_names = regular_settings_names
        key_cols = ["encoder_arch", "clf_arch", "feature_combine_type", "loss_type"]

    elif which_expt == "ablation":
        key_cols = [
            "encoder_arch",
            "clf_arch",
            "feature_combine_type",
            "loss_type",
            "linearity_loss_coeff",
            "real_CE_loss_coeff",
            "fake_CE_loss_coeff",
            "data_noise_SNR",
        ]
        settings_names = ablation_settings_names

    else:
        raise NotImplementedError()

    df = compute_similarities(
        settings_names, cols=key_cols + ["seed", "fold"], results_dir=results_dir, title=title, gamma=gamma
    )

    # Create table
    df.to_pickle(figs_dir / f"{title}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figs_dir", default="figures")
    parser.add_argument("--which_expt", required=True, choices=["regular", "ablation"])
    parser.add_argument("--gamma", default=None, type=float)
    args = parser.parse_args()
    results_dir = PROJECT_PATH / args.results_dir
    figs_dir = PROJECT_PATH / args.figs_dir
    figs_dir.mkdir(exist_ok=True, parents=True)
    main(results_dir, figs_dir, args.which_expt, args.gamma)
