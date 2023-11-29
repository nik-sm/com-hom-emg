"""Train fresh classifiers using test checkpoints"""
import argparse
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from ablation_settings import settings_names as ablation_settings_names
from loguru import logger
from pytorch_lightning import seed_everything
from regular_settings import settings_names as regular_settings_names
from rich.console import Console
from rich.table import Table
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from tqdm import tqdm
from utils import table_to_csv

from com_hom_emg.data import DataModule, get_per_subj_data
from com_hom_emg.model import LearnedEmbedding
from com_hom_emg.parallel_models import ControlModel_RandomGuess, ParallelA
from com_hom_emg.scoring import get_combo_conf_mat
from com_hom_emg.utils import PROJECT_PATH

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
    step = int(re.match(rf"{which}__epoch=\d+__step=(\d+)", chosen_ckpt.name).group(1))

    metrics = pd.read_csv(folder / "metrics.csv")
    results = {}

    # NOTE - for this experiment, we ignore the test results, which come from fine-tuning,
    # since we will train a fresh classifier instead
    for split in ["train", "val"]:
        cols = [col for col in metrics.columns if col.startswith(split)]
        if len(cols) == 0:
            raise FailedRunError(f"No {split} metrics found in {folder}")

        cols.append("step")
        subset = metrics[cols].dropna().set_index("step")
        subset = subset.iloc[subset.index.get_indexer([step], method="nearest")]
        assert len(subset) == 1
        results.update(**subset.to_dict(orient="records")[0])

    hparams = yaml.safe_load((folder / "hparams.yaml").read_text())

    return hparams, results, chosen_ckpt


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


def get_clf(name: str):
    if name == "logr":
        return LogR(class_weight="balanced", max_iter=4000, n_jobs=-1)
    elif name == "lda":
        return LDA()
    elif name == "knn":
        return KNN(n_jobs=-1)
    elif name == "rf":
        return RF(n_jobs=-1, class_weight="balanced")
    elif name == "dt":
        return DT(class_weight="balanced")
    else:
        raise ValueError(f"Unknown classifier name: {name}")


@torch.no_grad()
def try_fresh_classifier(embedding, test_loader, clf_name: str, test_frac=0.2, N_aug_each_class=500):
    # Get features
    embedding.to(device)
    features, labels, is_single = [], [], []
    for batch_data, batch_labels, batch_is_single, _subj_ids in test_loader:
        features.append(embedding(batch_data.to(device)))
        labels.append(batch_labels.to(device))
        is_single.append(batch_is_single)
    features = torch.cat(features)
    labels = torch.cat(labels)
    is_single = torch.cat(is_single)

    # Create a single train/test split
    N_single = is_single.sum().item()
    N_single_test = int(N_single * test_frac)

    N_double = (~is_single).sum().item()
    N_double_test = int(N_double * test_frac)

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

    # Define function to train a single sklearn clf
    def try_once(which: str):
        # logger.info(f"Train an example model for scenario: {which}")
        clf = ParallelA(dir_clf=get_clf(clf_name), mod_clf=get_clf(clf_name))
        control = ControlModel_RandomGuess()
        model = {"upper": clf, "lower": clf, "aug": clf, "control": control}[which]
        use_aug = {"upper": False, "lower": False, "aug": True, "control": False}[which]
        doubles_in_train = {"upper": True, "lower": False, "aug": False, "control": True}[which]

        if doubles_in_train:
            x_train = torch.cat((train_single_feat, train_double_feat))
            y_train = torch.cat((train_single_labels, train_double_labels))
        else:
            x_train = train_single_feat
            y_train = train_single_labels

        if use_aug:
            x_aug, y_aug = embedding.feature_combination(train_single_feat, train_single_labels)
            # logger.info(f"Real singles: {len(x_train)}, augmented: {len(x_aug)}")
            if N_aug_each_class is not None:
                x_aug, y_aug = subset_each_class(x_aug, y_aug, N_aug_each_class)
            # logger.info(f"Subset augmented: {len(x_aug)}")
            x_train = torch.cat([x_train, x_aug])
            y_train = torch.cat([y_train, y_aug])

        x_test = torch.cat([test_single_feat, test_double_feat])
        y_test = torch.cat([test_single_labels, test_double_labels])

        # After (possibly) applying augmentation fn - then we can convert to numpy
        # That way, augmentation fn can assume its input to be torch tensor
        x_train = x_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        x_test = x_test.cpu().numpy()
        y_test = y_test.cpu().numpy()
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        cm = get_combo_conf_mat(y_test, preds)
        cm_counts = get_combo_conf_mat(y_test, preds, normalize=False)

        single_bal_acc = np.nanmean(np.diag(cm)[:8])
        double_bal_acc = np.nanmean(np.diag(cm)[8:])
        overall_bal_acc = np.nanmean(np.diag(cm))
        return {
            "single_bal_acc": single_bal_acc,
            "double_bal_acc": double_bal_acc,
            "overall_bal_acc": overall_bal_acc,
            "confusion_matrix": cm,
            "confusion_matrix_counts": cm_counts,
        }

    return {
        # Train once with singles and doubles (upper-bound performance)
        "upper_bound": try_once("upper"),
        # Train once with only singles, no augmentation (lower-bound performance)
        "lower_bound": try_once("lower"),
        # Train once with singles only and augmentation
        "augmented": try_once("aug"),
        # Train once with a random model (lower-bound performance)
        # "control": try_once("control"),
    }


def fresh_classifier_one_ckpt(ckpt, clf_name: str, n_aug: Optional[int]):
    embedding = LearnedEmbedding.load_from_checkpoint(ckpt)
    embedding.eval()

    per_subj_data = get_per_subj_data()
    datamodule = DataModule(per_subj_data=per_subj_data, **embedding.hparams)

    results = try_fresh_classifier(embedding, datamodule.test_dataloader(), clf_name, N_aug_each_class=n_aug)
    # Flatten keys
    results = pd.json_normalize(results).iloc[0].to_dict()
    return results


def run_fresh_classifiers(
    settings: List[dict],
    cols: List[str],
    results_dir: Path,
    title: str,
    clf_name: str,
    n_aug: Optional[int],
):
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
                hparams, train_val_results, ckpt = load_one(folder)
            except FailedRunError:
                failed_runs.append(folder)
                continue
            fresh_test_results = fresh_classifier_one_ckpt(ckpt, clf_name=clf_name, n_aug=n_aug)

            # Construct a row, starting with hyperparams
            record = {k: hparams[k] for k in cols}

            # Add info from train and val
            # record.update(train_val_results)  # TODO - if train and val results are desired, include here

            # Add info from fresh test, setting aside confusion matrices
            record.update(fresh_test_results)

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


def make_table(df, figs_dir, key_names: List[str], key_cols: List[str], title: str):
    cols = []
    cols.extend(key_names + ["N"])
    cols.extend(["Aug_Sing", "Aug_Doub", "Aug_Ovr"])
    cols.extend(["Lower_Sing", "Lower_Doub", "Lower_Ovr"])
    cols.extend(["Upper_Sing", "Upper_Doub", "Upper_Ovr"])

    rows = []
    for key_vals, group_rows in df.groupby(key_cols, dropna=False):
        row = list(key_vals)

        runs_completed = len(group_rows)
        row.append(str(runs_completed))

        k1 = "augmented"
        for k2 in ["single_bal_acc", "double_bal_acc", "overall_bal_acc"]:
            vals = group_rows[f"{k1}.{k2}"]
            row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")

        k1 = "lower_bound"
        for k2 in ["single_bal_acc", "double_bal_acc", "overall_bal_acc"]:
            vals = group_rows[f"{k1}.{k2}"]
            row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")

        k1 = "upper_bound"
        for k2 in ["single_bal_acc", "double_bal_acc", "overall_bal_acc"]:
            vals = group_rows[f"{k1}.{k2}"]
            row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")

        rows.append(row)

    table = Table(title="Results")
    for col in cols:
        table.add_column(col)

    for row in rows:
        table.add_row(*[str(x) for x in row])

    console = Console(record=True, width=9999)
    console.print(table)
    console.save_text(figs_dir / f"{title}.txt")
    table_to_csv(table, figs_dir / f"{title}.csv")


@logger.catch(onerror=lambda _: sys.exit(1))
def main(results_dir: Path, figs_dir: Path, which_expt: str, clf_name: str, n_aug: Optional[int]):
    seed_everything(0)
    title = f"fresh-classifier.{which_expt}.{clf_name}.{n_aug}"
    if which_expt == "regular":
        settings_names = regular_settings_names
        key_names = ["encoder", "clf", "feat", "loss_type"]
        key_cols = ["encoder_arch", "clf_arch", "feature_combine_type", "loss_type"]

    elif which_expt == "ablation":
        key_names = ["encoder", "clf", "feat", "loss_type", "lin_loss", "real_CE", "fake_CE", "noise_SNR"]
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

    df = run_fresh_classifiers(
        settings_names,
        cols=key_cols + ["seed", "fold"],
        results_dir=results_dir,
        title=title,
        clf_name=clf_name,
        n_aug=n_aug,
    )

    # Create table
    make_table(df, figs_dir, key_names=key_names, key_cols=key_cols, title=title)
    df.to_pickle(figs_dir / f"{title}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figs_dir", default="figures")
    parser.add_argument("--which_expt", required=True, choices=["regular", "ablation"])
    parser.add_argument("--clf_name", default="rf")
    parser.add_argument("--n_aug", default=None, type=int, help="None to use all aug items, positive int to use subset")
    args = parser.parse_args()
    results_dir = PROJECT_PATH / args.results_dir
    figs_dir = PROJECT_PATH / args.figs_dir
    figs_dir.mkdir(exist_ok=True, parents=True)
    main(results_dir, figs_dir, args.which_expt, args.clf_name, args.n_aug)
