import itertools


def get_name(job):
    parts = []
    parts.append(f"fold={job['--fold']}")
    parts.append(f"seed={job['--seed']}")
    parts.append(f"encoder_arch={job['--encoder_arch']}")
    parts.append(f"clf_arch={job['--clf_arch']}")
    parts.append(f"feature_combine_type={job['--feature_combine_type']}")
    parts.append(f"loss_type={job['--loss_type']}")
    #
    parts.append(f"lin={job['--linearity_loss_coeff']}")
    parts.append(f"real={job['--real_CE_loss_coeff']}")
    parts.append(f"fake={job['--fake_CE_loss_coeff']}")
    noise = job.get("--data_noise_SNR", None)
    parts.append(f"noise={noise}")
    name = "__".join(parts)
    return name


fixed_args = {
    "--results_dir": ["results_ablation"],
    # Model:
    "--feature_dim": [64],
    "--normalized_features": [False],
    "--lr": [3e-4],
    "--lr_decay": [1.0],
    "--margin": [1.0],
    "--centroids_momentum": [0.75],
    "--triplets_per_item": [3],
    "--which_ckpt": ["best"],
    # Data:
    "--n_train_subj": [8],
    "--n_val_subj": [1],
    "--n_test_subj": [1],
    "--batch_size": [128],
    "--num_workers": [4],
    "--use_preprocessed_data": [False],
    # Trainer:
    "--enable_progress_bar": [False],
    "--accelerator": ["gpu"],
    "--devices": [1],
    "--finetune_steps": [10000],
    "--finetune_n_aug_per_class": [500],
    "--finetune_lr": [3e-5],
    "--finetune_lr_decay": [1.0],
    # # When testing:
    "--max_epochs": [300],
    # "--max_epochs": [1],
    # "--limit_train_batches": [1],
    # "--limit_val_batches": [1],
    # "--limit_test_batches": [1],
    # "--finetune_epochs": [1],
}

varying_args = {
    "--fold": range(10),
    "--seed": range(5),
    "--encoder_arch": ["basic"],  # "conformer", "vit"
    "--clf_arch": ["small"],
    "--loss_type": ["triplet"],
    "--feature_combine_type": ["mlp"],
}


ablation_settings = [
    # Varying lambdas:
    {"--linearity_loss_coeff": 0.0, "--real_CE_loss_coeff": 0.0, "--fake_CE_loss_coeff": 1.0, "--data_noise_SNR": 20},
    {"--linearity_loss_coeff": 0.0, "--real_CE_loss_coeff": 1.0, "--fake_CE_loss_coeff": 0.0, "--data_noise_SNR": 20},
    {"--linearity_loss_coeff": 0.0, "--real_CE_loss_coeff": 1.0, "--fake_CE_loss_coeff": 1.0, "--data_noise_SNR": 20},
    {"--linearity_loss_coeff": 1.0, "--real_CE_loss_coeff": 0.0, "--fake_CE_loss_coeff": 0.0, "--data_noise_SNR": 20},
    {"--linearity_loss_coeff": 1.0, "--real_CE_loss_coeff": 0.0, "--fake_CE_loss_coeff": 1.0, "--data_noise_SNR": 20},
    {"--linearity_loss_coeff": 1.0, "--real_CE_loss_coeff": 1.0, "--fake_CE_loss_coeff": 0.0, "--data_noise_SNR": 20},
    # Varying data_noise_SNR:
    {"--linearity_loss_coeff": 1.0, "--real_CE_loss_coeff": 1.0, "--fake_CE_loss_coeff": 1.0},
    # omit -> "None" -> no noise
    {"--linearity_loss_coeff": 1.0, "--real_CE_loss_coeff": 1.0, "--fake_CE_loss_coeff": 1.0, "--data_noise_SNR": 10},
    {"--linearity_loss_coeff": 1.0, "--real_CE_loss_coeff": 1.0, "--fake_CE_loss_coeff": 1.0, "--data_noise_SNR": 30},
]


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


combined_args = {**fixed_args, **varying_args}

settings = []
for setting_base in product_dict(**combined_args):
    for ablation_setting in ablation_settings:
        setting = {**setting_base, **ablation_setting}
        setting["--name"] = get_name(setting)
        settings.append(setting)
settings_names = [get_name(job) for job in settings]
