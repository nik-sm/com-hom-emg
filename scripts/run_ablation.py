from ablation_settings import settings
from utils import run_one

from com_hom_emg.utils import PROJECT_PATH

DRY_RUN = True

script = PROJECT_PATH / "com_hom_emg" / "main.py"
python = PROJECT_PATH / "venv" / "bin" / "python"
assert python.exists()

if DRY_RUN:
    print("#" * 80)
    print("DRY RUN")

running_job_count = 0
for job_kwargs in settings:
    job = f"{python} {script} "
    for flag, value in job_kwargs.items():
        job += f"{flag} {value} "
    run_one(job, running_job_count, dry_run=DRY_RUN)
    running_job_count += 1

print(f"Total jobs: {running_job_count}")
