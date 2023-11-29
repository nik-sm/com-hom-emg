import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

from com_hom_emg.scoring import CANONICAL_COORDS_STR
from com_hom_emg.utils import PROJECT_PATH

layout_template = "simple_white"
colors = px.colors.qualitative.Plotly


def plot_confusion_matrix(data: np.ndarray, title: Optional[str] = None):
    def make_text(cm):
        text = []
        for v in cm.flatten():
            text.append(f"{round(v, 2)}")
        return np.array(text).reshape(cm.shape)

    text = make_text(data)

    # Eliminate the final row, which corresponds to actual label = "None, None"
    data = data[:-1]
    text = text[:-1]

    ticktext = CANONICAL_COORDS_STR
    fig = go.Figure()
    showscale = False
    margin = dict(l=20, r=0, t=0, b=20)
    if title is not None:
        fig.update_layout(title=title)
        margin = dict(l=20, r=0, t=20, b=20)
    fig.add_trace(
        go.Heatmap(
            z=data,
            text=text,
            texttemplate="%{text}",
            zmin=0,
            zmax=1,
            colorscale="Blues",
            showscale=showscale,
            textfont_size=15,
        )
    )
    fig.update_layout(
        margin=margin,
        xaxis=dict(
            title="Predicted",
            tickangle=-45,
            tickmode="array",
            ticktext=ticktext,
            tickvals=list(range(len(ticktext))),
            constrain="domain",
        ),
        yaxis=dict(
            title="Actual",
            tickmode="array",
            ticktext=ticktext,
            tickvals=list(range(len(ticktext))),
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        font_size=15,
    )

    full_fig = fig.full_figure_for_development(warn=False)
    x_lo, x_hi = full_fig.layout.xaxis.range
    y_hi, y_lo = full_fig.layout.yaxis.range  # NOTE - y-axis range is reversed for heatmap
    box_size = (y_hi - y_lo) / data.shape[0]

    # Add horizontal line between single and combo classes
    n = 8  # 8 single classes above the line
    x = [x_lo, x_hi]
    y_value = y_hi - n * box_size
    y = [y_value, y_value]
    y = [y_hi - y_ + y_lo for y_ in y]  # reverse coords
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", dash="dot", width=4), showlegend=False))

    # Add vertical line between single and combo classes
    n = 8  # 8 single classes left of the line
    x_value = x_lo + n * box_size
    x = [x_value, x_value]
    y = [y_hi, y_lo]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", dash="dot", width=4), showlegend=False))

    # Add verticla line between combo classes and 'None' class
    n = 24  # 24 classes left of the line
    x_value = x_lo + n * box_size
    x = [x_value, x_value]
    y = [y_hi, y_lo]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", dash="dot", width=4), showlegend=False))

    # Need to re-set the axis ranges after adding lines
    fig.update_layout(xaxis_range=[x_lo, x_hi], yaxis_range=[y_hi, y_lo], yaxis_autorange=False)
    return fig


def make_boxplots(df: pd.DataFrame, figs_dir, which_test: str, title: str):
    # Make 3 large figures: one on single acc, one on double acc, and one on overall acc
    # In each figure:
    # The figure contains grouped boxplots.
    # Each boxplot group is a particular setting of encoder, classifier, feature combine type, loss type
    # Within each boxplot group, we have a box for "augmented", "upper", "lower". In the case of fine-tuning,
    #   we also have a box for "zero-shot"

    # First, unify column naming. In fine-tune, names are "test_augmented/overall_bal_acc", etc
    # In fresh-classifier, names are "augmented.overall_bal_acc", etc
    # Stick to the latter.

    output_dir = figs_dir / f"{title}.boxplots"
    output_dir.mkdir(exist_ok=True)

    for subset in ["single", "double", "overall"]:
        fig = go.Figure()

        names_cols = [
            ("augmented", f"augmented.{subset}_bal_acc"),
            ("upper_bound", f"upper_bound.{subset}_bal_acc"),
            ("lower_bound", f"lower_bound.{subset}_bal_acc"),
        ]
        if which_test == "finetune":
            names_cols.append(("zero_shot", f"zero_shot.{subset}_bal_acc"))

        for i, (name, col) in enumerate(names_cols):
            data = df[col]
            x = df["group_name"]
            kw = dict(jitter=0.5, marker_size=3, marker_color=colors[i])
            trace = go.Box(y=data, x=x, name=name, **kw)
            fig.add_trace(trace)

        fig.update_layout(
            boxmode="group",
            template=layout_template,
            yaxis=dict(range=[0, 1], title="Balanced Test Acc"),
            xaxis_title="Classifier // Feature Combine Type // Loss Type",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            # boxgap=0.25,  # Space between groups
            # boxgroupgap=0,  # Space between boxes in a group
            margin=dict(l=0, r=0, t=0, b=0),
            font_size=15,
        )
        fig.write_image(output_dir / f"{title}.{subset}.png", width=1200, height=600, scale=2)


def make_confusion_matrices(df: pd.DataFrame, figs_dir, which_test: str, title: str):
    # Group by method details, and then average across folds and seeds
    # Create a single confusion matrix using plot_confusion_matrix
    # Save to file
    output_dir = figs_dir / f"{title}.confusion_matrices"
    output_dir.mkdir(exist_ok=True)

    names = ["augmented", "upper_bound", "lower_bound"]
    if which_test == "finetune":
        names.append("zero_shot")

    for group_name, group in df.groupby("group_name"):
        for name in names:
            col = f"{name}.confusion_matrix"
            this_group_conf_mats = np.stack(group[col])
            avg_conf_mat = np.nanmean(this_group_conf_mats, 0)
            fig = plot_confusion_matrix(avg_conf_mat)  # , title=f"{group_name} // {name}")
            filename = f"{title}.{group_name.replace('<br>', '__')}.{name}.conf_mat.png"
            fig.write_image(output_dir / filename, width=1000, height=1000, scale=2)


def plot_heatmap(data: np.ndarray, ticktext: List[str]):
    def make_text(cm):
        text = []
        for v in cm.flatten():
            text.append(f"{round(v, 2)}")
        return np.array(text).reshape(cm.shape)

    # Get lower triangular
    data = np.copy(data)
    data[np.triu_indices(data.shape[0], k=1)] = None

    text = make_text(data)

    fig = go.Figure()
    fig.update_layout(
        # margin=margin,
        template=layout_template,
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            ticktext=ticktext,
            tickvals=list(range(len(ticktext))),
            constrain="domain",
        ),
        yaxis=dict(
            tickmode="array",
            ticktext=ticktext,
            tickvals=list(range(len(ticktext))),
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        width=1000,
        height=1000,
        margin=dict(l=0, r=0, t=0, b=0),
        font_size=15,
    )
    fig.add_trace(
        go.Heatmap(z=data, text=text, texttemplate="%{text}", zmin=0, zmax=1, colorscale="Greens", showscale=False)
    )
    return fig


def make_similarity_heatmap_plot(similarity_matrix, ticktext):
    fig = plot_heatmap(similarity_matrix, ticktext)
    full_fig = fig.full_figure_for_development(warn=False)
    x_lo, x_hi = full_fig.layout.xaxis.range
    y_hi, y_lo = full_fig.layout.yaxis.range  # NOTE - y-axis range is reversed for heatmap
    n_classes = len(ticktext)
    box_size = (y_hi - y_lo) / n_classes

    # Add a line after the single gesture classes
    def add_hline(n):
        # Line from the y-axis, travling horizontall, until it hits the diagonal
        x = [x_lo, x_lo + n * box_size]
        # compute y-values in the normal way
        y = [y_hi - n * box_size, y_hi - n * box_size]
        # Then adjust y values to account for reversed axis
        y = [y_hi - y_ + y_lo for y_ in y]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", dash="dot", width=4), showlegend=False)
        )

    def add_vline(n):
        # Line from the diagonal, traveling vertically down, until it hits x-axis
        # after moving over n boxes, the y value of the diagonal is
        x = [x_lo + n * box_size, x_lo + n * box_size]
        # compute y-values in the normal way
        y = [y_hi - n * box_size, y_lo]
        # Then adjust y values to account for reversed axis
        y = [y_hi - y_ + y_lo for y_ in y]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", dash="dot", width=4), showlegend=False)
        )

    # Add lines for easier interpretation
    # p fig.full_figure_for_development(warn=False).layout.yaxis.range
    add_hline(16)
    add_vline(16)

    # Need to re-set the axis ranges after adding lines
    fig.update_layout(xaxis_range=[x_lo, x_hi], yaxis_range=[y_hi, y_lo], yaxis_autorange=False)
    return fig


def summarize_similarity_matrix(similarity_matrix: np.ndarray):
    # Extract 4 numbers of interest:
    # - avg of first 16 elements of diag -> describes real-real similarity
    # - avg of final 16 elements of diag -> fake-fake sim
    # - avg of 16th subdiagonal -> real-fake sim
    # - avg of all other below-diagonal elements -> non-matching sim

    real_real_sim = np.nanmean(np.diag(similarity_matrix)[:16])
    fake_fake_sim = np.nanmean(np.diag(similarity_matrix)[16:])
    real_fake_sim = np.nanmean(np.diag(similarity_matrix, k=-16))

    # We want to get the avg of below-diagonal entries, except for a certain subdiagonal.
    # Add them all up, subtract that subdiagonal, and divide by number of items
    tril = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)]
    stripe = np.diag(similarity_matrix, k=-16)
    nonmatch_sim = (np.nansum(tril) - np.nansum(stripe)) / (len(tril) - len(stripe))
    return real_real_sim, fake_fake_sim, real_fake_sim, nonmatch_sim


def make_feature_similarity_plots(df, figs_dir, which_expt, title: str, gamma: Optional[float]):
    # NOTE - For each fake double, we computed the median distance to matching real doubles
    # This gives us the median of ~40 distances for each point.
    # We have ~85K fake doubles total, each has 1 median.
    # Then we have 50 independent runs. Here we average over all 50*85K items
    output_dir = figs_dir / f"{which_expt}.similarity_matrices"
    output_dir.mkdir(exist_ok=True)

    print(f"Table describing feature similarity, for: {which_expt}")
    print()

    rows = []
    print("group_name, real_to_real, fake_to_fake, real_to_fake, non_matching")
    for group_name, group in df.groupby("group_name"):
        similarity_matrices = np.stack(group["similarity_matrix"])
        scalar_sim_values = [summarize_similarity_matrix(m) for m in similarity_matrices]
        real_reals = [s[0] for s in scalar_sim_values]
        fake_fakes = [s[1] for s in scalar_sim_values]
        real_fakes = [s[2] for s in scalar_sim_values]
        nonmatches = [s[3] for s in scalar_sim_values]

        real_to_real = f"{round(np.mean(real_reals), 2)} ± {round(np.std(real_reals), 2)}"
        fake_to_fake = f"{round(np.mean(fake_fakes), 2)} ± {round(np.std(fake_fakes), 2)}"
        real_to_fake = f"{round(np.mean(real_fakes), 2)} ± {round(np.std(real_fakes), 2)}"
        nonmatch = f"{round(np.mean(nonmatches), 2)} ± {round(np.std(nonmatches), 2)}"
        string = ", ".join([str(group_name), real_to_real, fake_to_fake, real_to_fake, nonmatch])
        print(string)
        rows.append(
            {
                "group_name": str(group_name),
                "real_to_real": real_to_real,
                "fake_to_fake": fake_to_fake,
                "real_to_fake": real_to_fake,
                "non_matching": nonmatch,
            }
        )
    print()
    # Save these summary statistics to a CSV so we can emit a latex table later
    table_df = pd.DataFrame(rows)
    table_df.to_csv(figs_dir / f"{title}.similarity_values.{gamma}.csv", index=False)

    print(f"Figures with average similarity heatmap for each group, for: {which_expt}")
    for group_name, group in df.groupby("group_name"):
        similarity_matrices = np.stack(group["similarity_matrix"])
        ticktext = group["ticktext"].iloc[0]
        avg_similarity_matrix = np.nanmean(similarity_matrices, 0)
        fig = make_similarity_heatmap_plot(avg_similarity_matrix, ticktext)
        filename = f"{which_expt}.{group_name.replace('<br>', '__')}.similarity_matrix.png"
        fig.write_image(output_dir / filename, scale=2)


def main(figs_dir: Path, which_test: str, which_expt: str, suffix: str, gamma: Optional[float]):
    logger.info(f"Saving figures to: {figs_dir}")
    if suffix is None:
        title = f"{which_test}.{which_expt}"
    else:
        title = f"{which_test}.{which_expt}.{suffix}"

    logger.info(f"Loading data for: {title}")
    df = pd.read_pickle(figs_dir / f"{title}.pkl")

    # Add group name for convenient grouping later
    logger.info("NOTE - not including encoder arch in group names (only used basic)")
    df["group_name"] = df["clf_arch"] + "<br>" + df["feature_combine_type"] + "<br>" + df["loss_type"]
    if which_expt == "ablation":
        df["group_name"] = (
            df["group_name"].astype(str)
            + "<br>("
            + df["linearity_loss_coeff"].astype(str)
            + ","
            + df["real_CE_loss_coeff"].astype(str)
            + ","
            + df["fake_CE_loss_coeff"].astype(str)
            + ","
            + df["data_noise_SNR"].astype(str)
            + ")"
        )

    # Unify column naming from fine-tuning and fresh-classifier experiments
    col_rename_map = {}
    for subset in ["single", "double", "overall"]:
        for scenario in ["augmented", "lower_bound", "upper_bound", "zero_shot"]:
            col_rename_map[f"test_{scenario}/{subset}_bal_acc"] = f"{scenario}.{subset}_bal_acc"

    df = df.rename(columns=col_rename_map)

    # Make plots
    make_confusion_matrices(df, figs_dir, which_test, title)
    # make_boxplots(df, figs_dir, which_test, title)

    # NOTE - this part will get re-run a few times, but it is fine
    # (Because it doesn't depend on fine-tune vs fresh-classifier)
    # As long as this script is run once for "regular" and once for "ablation", it is enough
    df = pd.read_pickle(figs_dir / f"feature_similarity.{which_expt}.{gamma}.pkl")
    df["group_name"] = df["clf_arch"] + "<br>" + df["feature_combine_type"] + "<br>" + df["loss_type"]
    if which_expt == "ablation":
        df["group_name"] = (
            df["group_name"].astype(str)
            + "<br>("
            + df["linearity_loss_coeff"].astype(str)
            + ","
            + df["real_CE_loss_coeff"].astype(str)
            + ","
            + df["fake_CE_loss_coeff"].astype(str)
            + ","
            + df["data_noise_SNR"].astype(str)
            + ")"
        )
    make_feature_similarity_plots(df, figs_dir, which_expt, title, gamma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs_dir", default="figures")
    parser.add_argument("--which_test", required=True, choices=["finetune", "fresh-classifier"])
    parser.add_argument("--which_expt", required=True, choices=["regular", "ablation"])
    parser.add_argument("--suffix", default=None)  # e.g. "lda.None" or "logr.1000"
    parser.add_argument("--gamma", default=None, type=float)

    args = parser.parse_args()
    if args.which_test == "fresh-classifier":
        if args.suffix is None:
            raise ValueError("Must specify suffix for fresh-classifier test")
    figs_dir = PROJECT_PATH / args.figs_dir
    figs_dir.mkdir(exist_ok=True, parents=True)
    main(figs_dir, args.which_test, args.which_expt, args.suffix, args.gamma)
