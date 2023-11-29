import subprocess
from pathlib import Path
from textwrap import dedent

import pandas as pd
from rich.table import Table
from rich.text import Text

from com_hom_emg.utils import PROJECT_PATH


def table_to_csv(rich_table: Table, filename: Path):
    def strip(item):
        return Text.from_markup(item).plain

    contents = {strip(x.header): [strip(y) for y in x.cells] for x in rich_table.columns}
    pd.DataFrame(contents).to_csv(filename)


def _run_one_slurm(inner_cmd: str, slurm_partition: str, slurm_logs_dir: Path, dry_run: bool):
    # TODO: determine minimum required memory
    wrapped_cmd = dedent(
        f"""
        sbatch
         --nodes=1
         --ntasks=1
         --cpus-per-task=5
         --time=8:00:00
         --job-name=ablation
         --partition={slurm_partition}
         --mem=16Gb
         --gres=gpu:v100-sxm2:1
         --output={slurm_logs_dir / "slurm-%j.out"}
         --open-mode=truncate
         --wrap=" {inner_cmd} "
        """
    ).replace("\n", " ")
    print(wrapped_cmd)
    if not dry_run:
        subprocess.run(wrapped_cmd, shell=True, check=True)


def _run_one_local(inner_cmd: str, global_job_count: int, dry_run):
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
    if not dry_run:
        subprocess.run(wrapped_cmd, shell=True, check=True)


ON_SLURM_CLUSTER = subprocess.run("which sbatch", shell=True, check=False, capture_output=True).returncode == 0
if ON_SLURM_CLUSTER:
    slurm_logs_dir = PROJECT_PATH / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True, parents=True)


def run_one(job: str, running_job_count: int, dry_run: bool):
    if ON_SLURM_CLUSTER:
        slurm_partition = "multigpu"
        _run_one_slurm(job, slurm_partition, slurm_logs_dir, dry_run)
    else:
        _run_one_local(job, running_job_count, dry_run)
