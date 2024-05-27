from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import plotly.graph_objects as go
import umap
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

from com_hom_emg.data import get_per_subj_data
from com_hom_emg.utils import DIRECTION_GESTURES, MODIFIER_GESTURES, PROJECT_PATH


def try_once(data, labels, seed):
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, random_state=seed)
    model = RF(n_jobs=-1, class_weight="balanced")
    model.fit(train_x, train_y)
    train_preds = model.predict(train_x)
    train_bal_acc = balanced_accuracy_score(train_y, train_preds)
    test_preds = model.predict(test_x)
    test_bal_acc = balanced_accuracy_score(test_y, test_preds)
    return train_bal_acc, test_bal_acc


def sanity_check(data, dir_labels, mod_labels):
    idx = np.arange(len(data))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=0)

    train_data = data[train_idx]
    test_data = data[test_idx]

    train_dir_labels = dir_labels[train_idx]
    train_mod_labels = mod_labels[train_idx]

    test_dir_labels = dir_labels[test_idx]
    test_mod_labels = mod_labels[test_idx]

    dir_model = RF(n_jobs=-1, class_weight="balanced")
    dir_model.fit(train_data, train_dir_labels)
    dir_preds = dir_model.predict(test_data)

    mod_model = RF(n_jobs=-1, class_weight="balanced")
    mod_model.fit(train_data, train_mod_labels)
    mod_preds = mod_model.predict(test_data)

    dir_correct = dir_preds == test_dir_labels
    mod_correct = mod_preds == test_mod_labels

    dir_test_acc = dir_correct.sum() / len(test_data)
    mod_test_acc = mod_correct.sum() / len(test_data)

    exact_test_acc = (dir_correct & mod_correct).sum() / len(test_data)

    print(f"Dir test acc: {dir_test_acc:.3f}. Mod test acc: {mod_test_acc:.3f}. Exact match acc: {exact_test_acc:.3f}")


def make_umap_plot(data, labels):
    embed = umap.UMAP(verbose=False, n_jobs=-1)
    embed.fit(data)

    # Plot the same points twice. On top plot, color them by "single" vs "combo"
    # On bottom plot, color by class

    # Color points according to their labels. Classes of interest are:
    # Single gestures (Up, None) ... (Down, None)

    # First, subset to direction data only
    idx = labels[:, 0] != 4
    data = data[idx]
    labels = labels[idx]

    # Now make subplots.
    # First is colored with a binary label (single vs combo)
    is_single = labels[:, 1] == 4
    cols = np.where(is_single, "red", "blue")
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Scatter(
            x=embed.embedding_[:, 0],
            y=embed.embedding_[:, 1],
            mode="markers",
            marker=dict(size=5, color=cols),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=embed.embedding_[:, 0],
            y=embed.embedding_[:, 1],
            mode="markers",
            marker=dict(size=5, color=labels[:, 0]),
        ),
        row=2,
        col=1,
    )

    fig.show()
    breakpoint()


def main(args):
    results_dir = PROJECT_PATH / args.results_dir
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    per_subj_data = get_per_subj_data()

    keys = [
        "Single vs Combo",
        "Up+None vs Down+None",
        "Up+None vs Left+None",
        "Right+None vs Down+None",
        "Up+Pinch vs Down+Pinch",
        "Up+Some vs Down+Some",
        "Up+Any vs Down+Any",
        "Up+M vs D+Pinch",
        "Which Direction (5way)",
        "Which Direction (4way)",
        "Which Modifier (5way)",
        "Which Modifier (4way)",
    ]
    train_bal_accs = {k: [] for k in keys}
    test_bal_accs = {k: [] for k in keys}

    for subj, subj_data in tqdm(per_subj_data.items(), leave=True, desc="subjects"):
        # On each subject, train simple models to classify single gesture vs combo

        data, orig_labels = subj_data["data"], subj_data["labels"]
        data = data.reshape(len(data), -1)

        # make_umap_plot(data, orig_labels)

        is_none = (orig_labels[:, 0] == 4) & (orig_labels[:, 1] == 4)
        assert is_none.sum() == 0

        is_single = (orig_labels[:, 0] == 4) | (orig_labels[:, 1] == 4)
        assert is_single.sum() + (~is_single).sum() == len(data)
        # Now we know that data are well described by a binary label

        # sanity_check(data, orig_labels[:, 0], orig_labels[:, 1])

        for seed in trange(5, position=1, leave=False, desc="seeds"):
            name = "Single vs Combo"
            train_bal_acc, test_bal_acc = try_once(data, is_single, seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            # Up+None vs Down+None
            is_up_none = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Up")) & (orig_labels[:, 1] == 4)
            is_down_none = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Down")) & (orig_labels[:, 1] == 4)
            subset_idx = np.where(is_up_none | is_down_none)[0]
            name = "Up+None vs Down+None"
            train_bal_acc, test_bal_acc = try_once(data[subset_idx], is_up_none[subset_idx], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            # Up+None vs Left+None
            is_up_none = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Up")) & (orig_labels[:, 1] == 4)
            is_left_none = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Left")) & (orig_labels[:, 1] == 4)
            subset_idx = np.where(is_up_none | is_left_none)[0]
            name = "Up+None vs Left+None"
            train_bal_acc, test_bal_acc = try_once(data[subset_idx], is_up_none[subset_idx], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            # Up+Pinch vs Down+Pinch
            is_up_pinch = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Up")) & (
                orig_labels[:, 1] == MODIFIER_GESTURES.index("Pinch")
            )
            is_down_pinch = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Down")) & (
                orig_labels[:, 1] == MODIFIER_GESTURES.index("Pinch")
            )
            subset_idx = np.where(is_up_pinch | is_down_pinch)[0]
            name = "Up+Pinch vs Down+Pinch"
            train_bal_acc, test_bal_acc = try_once(data[subset_idx], is_up_pinch[subset_idx], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            # Right+None vs Down+None
            is_right_none = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Right")) & (orig_labels[:, 1] == 4)
            is_down_none = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Down")) & (orig_labels[:, 1] == 4)
            subset_idx = np.where(is_right_none | is_down_none)[0]
            name = "Right+None vs Down+None"
            train_bal_acc, test_bal_acc = try_once(data[subset_idx], is_right_none[subset_idx], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            # Up+Something vs Down+Something
            is_up_some = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Up")) & (orig_labels[:, 1] != 4)
            is_down_some = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Down")) & (orig_labels[:, 1] != 4)
            subset_idx = np.where(is_up_some | is_down_some)[0]
            name = "Up+Some vs Down+Some"
            train_bal_acc, test_bal_acc = try_once(data[subset_idx], is_up_some[subset_idx], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            # Up+Anything vs Down+Anything
            is_up = orig_labels[:, 0] == DIRECTION_GESTURES.index("Up")
            is_down = orig_labels[:, 0] == DIRECTION_GESTURES.index("Down")
            subset_idx = np.where(is_up | is_down)[0]
            name = "Up+Any vs Down+Any"
            train_bal_acc, test_bal_acc = try_once(data[subset_idx], is_up[subset_idx], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            # Up+Anything vs Anything+Pinch (Excluding Up+Pinch gestures)
            is_up_not_pinch = (orig_labels[:, 0] == DIRECTION_GESTURES.index("Up")) & (
                orig_labels[:, 1] != MODIFIER_GESTURES.index("Pinch")
            )
            is_pinch_not_up = (orig_labels[:, 1] == MODIFIER_GESTURES.index("Pinch")) & (
                orig_labels[:, 0] != DIRECTION_GESTURES.index("Up")
            )
            subset_idx = np.where(is_up_not_pinch | is_pinch_not_up)[0]
            name = "Up+M vs D+Pinch"
            train_bal_acc, test_bal_acc = try_once(data[subset_idx], is_up_not_pinch[subset_idx], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            name = "Which Direction (5way)"
            train_bal_acc, test_bal_acc = try_once(data, orig_labels[:, 0], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            name = "Which Direction (4way)"
            subset_idx = orig_labels[:, 0] != 4
            train_bal_acc, test_bal_acc = try_once(data[subset_idx], orig_labels[subset_idx][:, 0], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            name = "Which Modifier (5way)"
            train_bal_acc, test_bal_acc = try_once(data, orig_labels[:, 1], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

            name = "Which Modifier (4way)"
            subset_idx = orig_labels[:, 1] != 4
            train_bal_acc, test_bal_acc = try_once(data[subset_idx], orig_labels[subset_idx][:, 1], seed)
            train_bal_accs[name].append(train_bal_acc)
            test_bal_accs[name].append(test_bal_acc)

    for name in train_bal_accs.keys():
        print(name)
        train_mean = np.mean(train_bal_accs[name])
        train_std = np.std(train_bal_accs[name])
        print(f"\tTrain Bal Acc: {train_mean:.3f} ± {train_std:.3f}")

        test_mean = np.mean(test_bal_accs[name])
        test_std = np.std(test_bal_accs[name])
        print(f"\tTest Bal Acc:{test_mean:.3f} ± {test_std:.3f}")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument("--results_dir", default="results_supplementary")
    args = parser.parse_args()
    main(args)
