import warnings

import pandas as pd

from com_hom_emg.utils import PROJECT_PATH

warnings.filterwarnings("ignore")

print("Main tables")

figs_dir = PROJECT_PATH / "figures"
# NOTE - the choice of files here determines which classifier alg and n_aug is used
files = [
    "finetune.regular.csv",
    "finetune.ablation.csv",
    "fresh-classifier.regular.rf.500.csv",
    "fresh-classifier.ablation.rf.500.csv",
]
for file in files:
    df = pd.read_csv(figs_dir / file, index_col=False)
    df = df.iloc[:, 1:]  # Drop first col
    df = df.drop("encoder", "columns")

    n_cols = df.shape[1]
    table = df.to_latex(index=False, escape=True, column_format="|l" * n_cols + "|")
    table = table.replace("±", "$\\pm$")
    table = table.replace("\\\\", "\\\\ \\hline")

    print("-" * 80)
    print()
    print(file)
    print(table)
    print()


print("Table comparing choice of alg for fresh classifier")
files = [
    "fresh-classifier.regular.rf.500.csv",
    "fresh-classifier.regular.knn.500.csv",
    "fresh-classifier.regular.lda.500.csv",
    "fresh-classifier.regular.dt.500.csv",
    "fresh-classifier.regular.logr.500.csv",
]
selected = []
for file in files:
    df = pd.read_csv(figs_dir / file, index_col=False)
    df = df.iloc[:, 1:]  # Drop first col
    subset = df[(df["clf"] == "small") & (df["feat"] == "mlp") & (df["loss_type"] == "triplet")]
    subset.insert(0, "file", file)
    selected.append(subset)

df = pd.concat(selected)
df = df.drop(["encoder", "clf", "feat", "loss_type"], "columns")
n_cols = df.shape[1]
table = df.to_latex(index=False, escape=True, column_format="|l" * n_cols + "|")
table = table.replace("±", "$\\pm$")
table = table.replace("\\\\", "\\\\ \\hline")

print("-" * 80)
print()
print(file)
print(table)
print()

print("Table summarizing similarity matrices for different model hyperparams")
figs_dir = PROJECT_PATH / "figures"
# NOTE - the choice of files here determines which classifier alg and n_aug is used
files = [
    "fresh-classifier.regular.rf.500.similarity_values.0.0078125.csv",
    "fresh-classifier.ablation.rf.500.similarity_values.0.0078125.csv",
]
for file in files:
    df = pd.read_csv(figs_dir / file, index_col=False)

    n_cols = df.shape[1]
    table = df.to_latex(index=False, escape=True, column_format="|l" * n_cols + "|")
    table = table.replace("±", "$\\pm$")
    table = table.replace("\\\\", "\\\\ \\hline")

    print("-" * 80)
    print()
    print(file)
    print(table)
    print()
