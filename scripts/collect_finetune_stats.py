"""Collect results of fine-tuning classifiers from finished runs."""
import argparse
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.io as pio
import yaml
from ablation_settings import settings_names as ablation_settings_names
from loguru import logger
from regular_settings import settings_names as regular_settings_names
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from utils import table_to_csv

from com_hom_emg.utils import PROJECT_PATH

pio.templates.default = "simple_white"


class FailedRunError(Exception):
    pass


def load_one(folder: Path, which="best"):
    """
    Extract train, val, and test metrics from the specified checkpoint (best or last).
    (NOTE - be careful to specify the same checkpoint that was actually used for testing).

    Also extract hyperparams from hparams.yaml file.
    """

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

    # for split in ["val", "test"]:
    # NOTE - since the fine-tuning stage was re-run and saved as confusion matrices, we must now compute metrics from
    # those conf mats. In a future re-training, the "test" metrics could simply be used instead.
    # In that case, the column naming produced during training also needs to be fixed when loading
    # (e.g. "test_augmented/overall_bal_acc" -> "augmented.overall_bal_acc")
    for split in ["val"]:
        cols = [col for col in metrics.columns if col.startswith(split)]
        if len(cols) == 0:
            raise FailedRunError(f"No {split} metrics found in {folder}")

        cols.append("step")
        subset = metrics[cols].dropna().set_index("step")
        subset = subset.iloc[subset.index.get_indexer([step], method="nearest")]
        assert len(subset) == 1
        results.update(**subset.to_dict(orient="records")[0])

    # Load confusion matrices and add to results
    conf_mat_folder = folder / "ConfusionMatrices"
    if not conf_mat_folder.exists():
        raise FailedRunError(f"No confusion matrices found in {folder}")

    results["augmented.confusion_matrix"] = np.load(conf_mat_folder / "test.augmented.confusion_matrix.npy")
    results["upper_bound.confusion_matrix"] = np.load(conf_mat_folder / "test.upper_bound.confusion_matrix.npy")
    results["lower_bound.confusion_matrix"] = np.load(conf_mat_folder / "test.lower_bound.confusion_matrix.npy")
    results["zero_shot.confusion_matrix"] = np.load(conf_mat_folder / "test.zero_shot.confusion_matrix.npy")

    results["augmented.single_bal_acc"] = np.nanmean(np.diag(results["augmented.confusion_matrix"])[:8])
    results["augmented.double_bal_acc"] = np.nanmean(np.diag(results["augmented.confusion_matrix"])[8:])
    results["augmented.overall_bal_acc"] = np.nanmean(np.diag(results["augmented.confusion_matrix"]))

    results["upper_bound.single_bal_acc"] = np.nanmean(np.diag(results["upper_bound.confusion_matrix"])[:8])
    results["upper_bound.double_bal_acc"] = np.nanmean(np.diag(results["upper_bound.confusion_matrix"])[8:])
    results["upper_bound.overall_bal_acc"] = np.nanmean(np.diag(results["upper_bound.confusion_matrix"]))

    results["lower_bound.single_bal_acc"] = np.nanmean(np.diag(results["lower_bound.confusion_matrix"])[:8])
    results["lower_bound.double_bal_acc"] = np.nanmean(np.diag(results["lower_bound.confusion_matrix"])[8:])
    results["lower_bound.overall_bal_acc"] = np.nanmean(np.diag(results["lower_bound.confusion_matrix"]))

    results["zero_shot.single_bal_acc"] = np.nanmean(np.diag(results["zero_shot.confusion_matrix"])[:8])
    results["zero_shot.double_bal_acc"] = np.nanmean(np.diag(results["zero_shot.confusion_matrix"])[8:])
    results["zero_shot.overall_bal_acc"] = np.nanmean(np.diag(results["zero_shot.confusion_matrix"]))

    hparams = yaml.safe_load((folder / "hparams.yaml").read_text())

    return hparams, results


def compare_updated_stats(df):
    # NOTE - This function is only useful for comparing previous fine-tuning approach to new fine-tuning approach
    def compute_single_bal_acc(cm):
        return np.nanmean(np.diag(cm)[:8])

    def compute_double_bal_acc(cm):
        return np.nanmean(np.diag(cm)[8:])

    def compute_overall_bal_acc(cm):
        return np.nanmean(np.diag(cm))

    # Compare the accuracies loaded from metrics.csv, to the ones we can compute directly from the confusion matrix
    df["new_augmented_single_bal_acc"] = df["augmented.confusion_matrix"].apply(compute_single_bal_acc)
    df["new_augmented_double_bal_acc"] = df["augmented.confusion_matrix"].apply(compute_double_bal_acc)
    df["new_augmented_overall_bal_acc"] = df["augmented.confusion_matrix"].apply(compute_overall_bal_acc)

    df["new_upper_bound_single_bal_acc"] = df["upper_bound.confusion_matrix"].apply(compute_single_bal_acc)
    df["new_upper_bound_double_bal_acc"] = df["upper_bound.confusion_matrix"].apply(compute_double_bal_acc)
    df["new_upper_bound_overall_bal_acc"] = df["upper_bound.confusion_matrix"].apply(compute_overall_bal_acc)

    df["new_lower_bound_single_bal_acc"] = df["lower_bound.confusion_matrix"].apply(compute_single_bal_acc)
    df["new_lower_bound_double_bal_acc"] = df["lower_bound.confusion_matrix"].apply(compute_double_bal_acc)
    df["new_lower_bound_overall_bal_acc"] = df["lower_bound.confusion_matrix"].apply(compute_overall_bal_acc)

    df["new_zero_shot_single_bal_acc"] = df["zero_shot.confusion_matrix"].apply(compute_single_bal_acc)
    df["new_zero_shot_double_bal_acc"] = df["zero_shot.confusion_matrix"].apply(compute_double_bal_acc)
    df["new_zero_shot_overall_bal_acc"] = df["zero_shot.confusion_matrix"].apply(compute_overall_bal_acc)

    # Print the differences
    for k in ["single_bal_acc", "double_bal_acc", "overall_bal_acc"]:
        for split in ["augmented", "upper_bound", "lower_bound", "zero_shot"]:
            old_key = f"test_{split}/{k}"
            new_key = f"new_{split}_{k}"
            diff = df[new_key] - df[old_key]
            logger.info(f"{new_key} minus {old_key}: {diff.mean():.2f} ± {diff.std():.2f}")

    breakpoint()


def make_table(df, figs_dir, key_names: List[str], key_cols: List[str], title: str):
    cols = []
    cols.extend(key_names + ["N"])
    cols.extend(["Aug_Sing", "Aug_Doub", "Aug_Ovr"])
    cols.extend(["Upper_Sing", "Upper_Doub", "Upper_Ovr"])
    cols.extend(["Lower_Sing", "Lower_Doub", "Lower_Ovr"])
    cols.extend(["ZeroShot_Sing", "ZeroShot_Doub", "ZeroShot_Ovr"])

    rows = []
    for key_vals, group_rows in df.groupby(key_cols):
        row = list(key_vals)

        runs_completed = len(group_rows)
        row.append(str(runs_completed))

        # Add aug cols
        k1 = "augmented"
        for k2 in ["single_bal_acc", "double_bal_acc", "overall_bal_acc"]:
            vals = group_rows[f"{k1}.{k2}"]
            row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")

        # Add upper bound cols
        k1 = "upper_bound"
        for k2 in ["single_bal_acc", "double_bal_acc", "overall_bal_acc"]:
            vals = group_rows[f"{k1}.{k2}"]
            row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")

        # Add lower bound cols
        k1 = "lower_bound"
        for k2 in ["single_bal_acc", "double_bal_acc", "overall_bal_acc"]:
            vals = group_rows[f"{k1}.{k2}"]
            row.append(f"{vals.mean():.2f} ± {vals.std():.2f}")

        # Add zero shot cols
        k1 = "zero_shot"
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


def process_one(settings: List[str], cols: List[str], results_dir: Path, title: str):
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
                continue
            settings.pop(idx)  # Now that we've found this setting, remove from the list

            # Try to load this setting
            try:
                hparams, results = load_one(folder)
            except FailedRunError:
                failed_runs.append(folder)
                continue
            record = {k: hparams[k] for k in cols}
            record.update(results)
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
def main(results_dir: Path, figs_dir: Path, which_expt: str):
    title = f"finetune.{which_expt}"
    if which_expt == "regular":
        key_names = ["encoder", "clf", "feat", "loss_type"]
        key_cols = ["encoder_arch", "clf_arch", "feature_combine_type", "loss_type"]
        settings_names = regular_settings_names

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

    df = process_one(settings_names, cols=key_cols + ["seed", "fold"], results_dir=results_dir, title=title)
    # compare_updated_stats(df)
    make_table(df, figs_dir, key_names=key_names, key_cols=key_cols, title=title)
    df.to_pickle(figs_dir / f"{title}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figs_dir", default="figures")
    parser.add_argument("--which_expt", required=True, choices=["regular", "ablation"])
    args = parser.parse_args()
    results_dir = PROJECT_PATH / args.results_dir
    figs_dir = PROJECT_PATH / args.figs_dir
    figs_dir.mkdir(exist_ok=True, parents=True)
    main(results_dir, figs_dir, args.which_expt)
