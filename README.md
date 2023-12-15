Code for **"Fast and Expressive Gesture Recognition using a Combination-Homomorphic Electromyogram Encoder"** by Niklas Smedemark-Margulies, Yunus Bicer, Elifnur Sunger, Tales Imbiriba, Eugene Tunik, Deniz Erdogmus, Mathew Yarossi, and Robin Walters

# Setup

Setup project with `make` and activate virtual environment with `source venv/bin/activate`.

### Download dataset

Download and uncompress the dataset from Zenodo:
```shell
bash scripts/fetch_dataset.sh
```

# Usage

The experiment workflow is divided into 3 stages:
1. Model training
2. Evaluation and collecting results
3. Plotting

### 1. Run training

Each experiment involves training models with varying hyperparameters, data splits, and random seeds.
Models are trained on data from a subset of subjects, and then the saved model can be used for feature extraction on 
unseen test subjects.

In the paper, we only discuss evaluation on unseen subjects using fresh small classifiers in scikit-learn.

NOTE:
- The training process includes an auxiliary classifier, and this repo includes code for fine-tuning that auxiliary classifier on the unseen test subjects (instead of training a fresh classifier).
- The launcher scripts here start with `DRY_RUN=True`, so that the proposed jobs can be inspected.
  Change this line to `DRY_RUN=False` to actually run the jobs.

Launch training runs using:
```shell
source venv/bin/activate
python scripts/run.py           # train models and get fine-tuning metrics
python scripts/run_ablation.py  # train ablated models
```

### 2. Evaluate models and collect results

Load checkpoints, train fresh classifiers, and evaluate on unseen test subjects using:
```shell
python scripts/collect_fresh_classifier_stats.py --results_dir results_regular --which_expt regular --n_aug 500 --clf_name rf
python scripts/collect_fresh_classifier_stats.py --results_dir results_ablation --which_expt ablation --n_aug 500 --clf_name rf
```

Note that if some training runs failed or cannot be loaded, the scripts above will create a list of failed runs.
These runs can be re-launched using:
```shell
python scripts/retry_regular_runs.py
python scripts/retry_ablation_runs.py
```

Load checkpoints, and compute feature similarities:
```shell
# NOTE: 1/128 = 0.0078125, the expected L2 distance between two vectors from a unit Gaussian in 64 dimensions
python scripts/compute_feature_similarity.py --results_dir results_regular --which_expt regular --gamma 0.0078125
python scripts/compute_feature_similarity.py --results_dir results_ablation --which_expt ablation --gamma 0.0078125
```

To collect results from using a fine-tuned classifier rather than a freshly trained one (not included in paper), use:
```shell
## Get results from fine-tuning - see notes above
# python scripts/collect_finetune_stats.py --results_dir results_regular --which_expt regular
# python scripts/collect_finetune_stats.py --results_dir results_ablation --which_expt ablation
```

### 3. Create plots and tables

Create plots and print out tables using:
```shell
python scripts/plots.py --which_test fresh-classifier --which_expt regular --suffix rf.500 --gamma 0.0078125
python scripts/plots.py --which_test fresh-classifier --which_expt ablation --suffix rf.500 --gamma 0.0078125
```

If interested in plots from fine-tuning results, use these instead:
```shell
# python scripts/plots.py --which_test finetune --which_expt regular
# python scripts/plots.py --which_test finetune --which_expt ablation
```

Draft latex tables using:
```shell
# NOTE - hard-coded file names in this script should be updated if changing options in previous steps such as `--gamma`
python scripts/emit_latex.py
```

# Dataset details

For information on the dataset contents, see the paper or the description at Zenodo: https://zenodo.org/records/10359729.

Data used in this project was recorded and preprocessed using the GEST project.

The dataset consists of 10 subjects, whose data were collected in the "Sprint7" experimental paradigm of the GEST project.

To produce the self-contained data file used here:
1. Checkout this branch of GEST repo: https://github.com/neu-spiral/GEST/tree/combination-homomorphic-encoder-dataset, and see the README at `reporting/`

2. Extract data from each subject's experiment folder using: `python reporting/sprint7/1_preprocess.py`
3. Export data: `python reporting/public-dataset-release/prepare-data.py`

# PDF

To read our paper, see: https://arxiv.org/pdf/2311.14675.pdf

# Citation

If you use this code or dataset, please use one of the citations below.

Article Citation:
```bibtex
@article{smedemarkmargulies2023fast,
  title={{Fast and Expressive Gesture Recognition using a Combination-Homomorphic Electromyogram Encoder}}, 
  author={
    Niklas Smedemark-Margulies and 
    Yunus Bicer and 
    Elifnur Sunger and 
    Tales Imbiriba and 
    Eugene Tunik and 
    Deniz Erdogmus and 
    Mathew Yarossi and 
    Robin Walters
  },
  year={2023},
  month={10},
  day={30},
  journal={arXiv preprint arXiv:2311.14675},
  url={https://arxiv.org/abs/2311.14675},
}
```

Dataset Citation:
```bibtex
@dataset{smedemarkmargulies_2023_10359729,
  title={{EMG from Combination Gestures following Visual Prompts}},
  author={
    Niklas Smedemark-Margulies and
    Yunus Bicer and
    Elifnur Sunger and
    Tales Imbiriba and
    Eugene Tunik and
    Deniz Erdogmus and
    Mathew Yarossi and
    Robin Walters
  },
  year={2023},
  month={12},
  day={11},
  publisher={Zenodo},
  version={1.0.1},
  doi={10.5281/zenodo.10359729},
  url={https://doi.org/10.5281/zenodo.10359729}
}
```
