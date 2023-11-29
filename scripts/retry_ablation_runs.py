import itertools
import re
from copy import deepcopy

from ablation_settings import fixed_args, get_name
from utils import run_one

from com_hom_emg.utils import PROJECT_PATH

DRY_RUN = True

# From each failed run, extract a dictionary with the necessary variables:
#   fold,seed, encoder_arch, clf_arch, loss_type, feature_combine_type,
#   linearity_loss_coeff, real_CE_loss_coeff, fake_CE_loss_coeff, data_noise_SNR
# using the following template:
# date=2023-03-06__ablation__fold=0__seed=0__encoder_arch=basic__clf_arch=small__feature_combine_type=mlp__loss_type=triplet__lin=0.0__real=1.0__fake=0.0__noise=20  # noqa
failed_settings = []
failed_runs_file = PROJECT_PATH / "FAILED_RUNS.finetune.ablation.txt"
missing_runs_file = PROJECT_PATH / "MISSING_RUNS.ablation.txt"
contents = []
if failed_runs_file.exists():
    contents.extend(failed_runs_file.read_text().splitlines())
if missing_runs_file.exists():
    contents.extend(missing_runs_file.read_text().splitlines())
for line in contents:
    fold = re.search(r"fold=([0-9])__", line).group(1)
    seed = re.search(r"seed=([0-4])__", line).group(1)
    encoder_arch = re.search(r"encoder_arch=(basic|conformer)__", line).group(1)
    clf_arch = re.search(r"clf_arch=(small|large)__", line).group(1)
    feature_combine_type = re.search(r"feature_combine_type=(avg|mlp)__", line).group(1)
    loss_type = re.search(r"loss_type=(triplet|triplet-hard|triplet-centroids)__", line).group(1)

    linearity_loss_coeff = re.search(r"lin=(0.0|1.0)__", line).group(1)
    real_CE_loss_coeff = re.search(r"real=(0.0|1.0)__", line).group(1)
    fake_CE_loss_coeff = re.search(r"fake=(0.0|1.0)__", line).group(1)
    data_noise_SNR = re.search(r"noise=([0-9]+|None)", line).group(1)

    assert int(fold) in range(10)
    assert int(seed) in range(5)
    assert encoder_arch in ["basic"]
    assert clf_arch in ["small", "large"]
    assert feature_combine_type in ["avg", "mlp"]
    assert loss_type in ["triplet", "triplet-hard", "triplet-centroids"]

    assert linearity_loss_coeff in ["0.0", "1.0"]
    assert real_CE_loss_coeff in ["0.0", "1.0"]
    assert fake_CE_loss_coeff in ["0.0", "1.0"]
    assert data_noise_SNR in ["None", "10", "20", "30"]

    setting = {
        "--fold": int(fold),
        "--seed": int(seed),
        "--encoder_arch": encoder_arch,
        "--clf_arch": clf_arch,
        "--feature_combine_type": feature_combine_type,
        "--loss_type": loss_type,
        "--linearity_loss_coeff": float(linearity_loss_coeff),
        "--real_CE_loss_coeff": float(real_CE_loss_coeff),
        "--fake_CE_loss_coeff": float(fake_CE_loss_coeff),
    }
    if data_noise_SNR != "None":
        setting["--data_noise_SNR"] = int(data_noise_SNR)

    failed_settings.append(setting)

print(f"Found {len(failed_settings)} failed runs.")

ALL_ARGS = fixed_args

################################################################################

script = PROJECT_PATH / "com_hom_emg" / "main.py"
python = PROJECT_PATH / "venv" / "bin" / "python"
assert python.exists()


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


if DRY_RUN:
    print("#" * 80)
    print("DRY RUN")


# Check if `sbatch` command is available in environment
running_job_count = 0
for job_kwargs_base in product_dict(**ALL_ARGS):
    for setting in failed_settings:
        job_kwargs = deepcopy(job_kwargs_base)
        job_kwargs.update(setting)
        job_kwargs["--name"] = get_name(job_kwargs)
        job = f"{python} {script} "
        for flag, value in job_kwargs.items():
            job += f"{flag} {value} "
        run_one(job, running_job_count, dry_run=DRY_RUN)
        running_job_count += 1

print(f"Total jobs: {running_job_count}")
