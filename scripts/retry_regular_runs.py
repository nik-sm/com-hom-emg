import itertools
import re
import subprocess
from copy import deepcopy
from textwrap import dedent

from regular_settings import fixed_args, get_name

from com_hom_emg.utils import PROJECT_PATH

DRY_RUN = True

# From each failed run, extract a dictionary with the necessary variables:
#   fold,seed, encoder_arch, clf_arch, loss_type, feature_combine_type
# using the following template:
# /home/niklas/git/learned-emg-encoder/results_2023-03-06/date=2023-03-06__fold=9__seed=4__encoder_arch=basic__clf_arch=large__feature_combine_type=avg__loss_type=triplet  # noqa
failed_settings = []
failed_runs_file = PROJECT_PATH / "FAILED_RUNS.regular.txt"
missing_runs_file = PROJECT_PATH / "MISSING_RUNS.regular.txt"
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
    loss_type = re.search(r"loss_type=(triplet|triplet-hard|triplet-centroids)$", line).group(1)

    assert int(fold) in range(10)
    assert int(seed) in range(5)
    assert encoder_arch in ["basic"]
    assert clf_arch in ["small", "large"]
    assert feature_combine_type in ["avg", "mlp"]
    assert loss_type in ["triplet", "triplet-hard", "triplet-centroids"]

    failed_settings.append(
        {
            "--fold": int(fold),
            "--seed": int(seed),
            "--encoder_arch": encoder_arch,
            "--clf_arch": clf_arch,
            "--feature_combine_type": feature_combine_type,
            "--loss_type": loss_type,
        }
    )

print(f"Found {len(failed_settings)} failed runs.")

ALL_ARGS = fixed_args

################################################################################

script = PROJECT_PATH / "com_hom_emg" / "main.py"
python = PROJECT_PATH / "venv" / "bin" / "python"
results_dir = PROJECT_PATH / "results"
slurm_logs_dir = results_dir / "slurm_logs"
slurm_logs_dir.mkdir(exist_ok=True, parents=True)
assert python.exists()


def run_one_slurm(slurm_partition: str, inner_cmd: str):
    # TODO: determine minimum required memory
    wrapped_cmd = dedent(
        f"""
        sbatch
         --nodes=1
         --ntasks=1
         --cpus-per-task=5
         --time=8:00:00
         --job-name=com_hom_emg
         --partition={slurm_partition}
         --mem=16Gb
         --gres=gpu:v100-sxm2:1
         --output={slurm_logs_dir / "slurm-%j.out"}
         --open-mode=truncate
         --wrap=" {inner_cmd} "
        """
    ).replace("\n", " ")
    print(wrapped_cmd)
    if not DRY_RUN:
        subprocess.run(wrapped_cmd, shell=True, check=True)


def run_one_local(inner_cmd: str, global_job_count: int):
    # use sem to limit the number of concurrent jobs
    ALLOWED_CUDA_DEVICE_IDS = [0, 1]
    cuda_device_id = ALLOWED_CUDA_DEVICE_IDS[global_job_count % len(ALLOWED_CUDA_DEVICE_IDS)]

    wrapped_cmd = dedent(
        f"""
        CUDA_VISIBLE_DEVICES={cuda_device_id}
         sem --id {cuda_device_id} --jobs 1
         {inner_cmd}
         >/dev/null 2>&1 &
        """
    ).replace("\n", " ")
    print(wrapped_cmd)
    if not DRY_RUN:
        subprocess.run(wrapped_cmd, shell=True, check=True)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


if DRY_RUN:
    print("#" * 80)
    print("DRY RUN")


# Check if `sbatch` command is available in environment
ON_SLURM_CLUSTER = subprocess.run("which sbatch", shell=True, check=False, capture_output=True).returncode == 0
n_total_jobs = 0
for job_kwargs_base in product_dict(**ALL_ARGS):
    for setting in failed_settings:
        job_kwargs = deepcopy(job_kwargs_base)
        job_kwargs.update(setting)
        job_kwargs["--name"] = get_name(job_kwargs)
        job = f"{python} {script} "
        for flag, value in job_kwargs.items():
            job += f"{flag} {value} "
        if ON_SLURM_CLUSTER:
            slurm_partition = "multigpu"
            run_one_slurm(slurm_partition, job)
        else:
            run_one_local(job, n_total_jobs)
        n_total_jobs += 1

print(f"Total jobs: {n_total_jobs}")
