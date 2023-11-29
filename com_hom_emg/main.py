"""
Given N subjects, use (N-1) subjects to learn a feature embedding that allows linear combination for data augmentation.
Evaluate its performance on the Nth subject by using it to train a small classifier on augmented data.
A useful feature embedding would allow the classifier to generalize well to unseen combination gestures after training
on only single gestures and fake combinations (created using linear combinations in the learned feature space)

1. Split subjects

2. Learn feature embedding on N-1 subjects.
    - Split data into single gestures and combination gestures.
    - Define feature embedding F (probably a small convolutional network).
    - Choose some notion of feature space distance D (probably L2 or RBF).
    - Define a contrastive-learning style loss function to achieve two properties:
        (1) minimize the distance between a linear combination of singles, and the corresponding double.
        (2) maximize distance between the same linear combo and a random unrelated double.

        fake_double = F(Up) + F(Pinch)
        positive_example = F(Up&Pinch)
        negative_example = F(Left&Thumb)
        term1 = D(fake_double, positive_example)  # Make the fake double close to the real double
        term2 = -D(fake_double, negative_example)  # Make the fake double far from the random double
        L = max(0, )
        L = term1 + term2

3. Train a simple classifier for the Nth subject, using the learned feature embedding to create augmented data
    - Split data into train (single gestures) and test (some single gestures and some combination gestures).
    - Create fake combination gestures from real single gestures
    - Train classifier and evaluate on test set
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from pprint import pformat

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

from com_hom_emg.data import DataModule, get_per_subj_data
from com_hom_emg.model import LearnedEmbedding
from com_hom_emg.utils import PROJECT_PATH


def main(args):
    pl.seed_everything(args.seed)
    results_dir = PROJECT_PATH / args.results_dir
    results_dir.mkdir(exist_ok=True, parents=True)
    per_subj_data = get_per_subj_data()

    # Setup trainer
    tb_logger = TensorBoardLogger(
        results_dir,
        name=None,
        version=args.name,
        log_graph=True,
        default_hp_metric=False,
        flush_secs=10,
    )
    csv_logger = CSVLogger(results_dir, name=None, version=tb_logger.version)
    if not args.fast_dev_run:
        logger.add(Path(tb_logger.log_dir) / "log.txt")
    # checkpointing
    # monitor = "val_augmented/overall_bal_acc"
    monitor = "val/overall_bal_acc"
    fname_parts = [
        "epoch={epoch}__",
        "step={step}__",
        # f"val_aug_overall_acc={{{monitor}:.3f}}",
        f"val_overall_bal_acc={{{monitor}:.3f}}",
    ]
    filename = "".join(fname_parts)
    best_ckpt = ModelCheckpoint(
        save_top_k=1, monitor=monitor, mode="max", filename="best__" + filename, auto_insert_metric_name=False
    )
    last_ckpt = ModelCheckpoint(filename="last__" + filename, auto_insert_metric_name=False)
    trainer = Trainer.from_argparse_args(
        args,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        deterministic=True,
        callbacks=[LearningRateMonitor(), best_ckpt, last_ckpt],
        logger=[tb_logger, csv_logger],
    )

    # Setup data
    datamodule = DataModule(per_subj_data=per_subj_data, **vars(args))

    # Setup model
    learned_embedding = LearnedEmbedding(
        input_channels=datamodule.example_data_shape[0],
        input_time_length=datamodule.example_data_shape[1],
        **vars(args),
    )
    logger.info(f"Model: {learned_embedding}")
    logger.info(f"Model summary: {ModelSummary(learned_embedding)}")

    # Train
    logger.info("Begin training...")
    logger.info(f"Args: {pformat(vars(args))}")

    if args.run_lr_tune:
        lr_finder = trainer.tuner.lr_find(
            learned_embedding,
            train_dataloaders=datamodule.train_dataloader(),
            num_training=1000,
        )

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder.png")
        logger.info("LR FINDER", lr_finder.suggestion())
        breakpoint()

    trainer.fit(learned_embedding, datamodule=datamodule, ckpt_path=args.resume_ckpt_path)
    logger.info("Finished training.")

    # Test
    if args.which_ckpt == "best":
        finetune_evaluation = trainer.test(ckpt_path="best", datamodule=datamodule)
    elif args.which_ckpt == "last":
        finetune_evaluation = trainer.test(learned_embedding, datamodule=datamodule)
    else:
        raise ValueError(f"Invalid choice: {args.which_ckpt=}")
    # Save results
    logger.info(f"Evaluation using fine-tuning:\n{pformat(finetune_evaluation)}")


if __name__ == "__main__":
    # NOTE - we send all args to all constructors. Thus:
    # - must not have name collision for args in model vs data vs trainer
    # - name of argparse arg must match name of arg in init fn
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default=None, help="if missing, uses version_0, version_1, etc")
    parser.add_argument("--run_lr_tune", action="store_true", default=False)
    parser.add_argument("--which_ckpt", choices=["best", "last"], default="best")
    parser.add_argument("--resume_ckpt_path", type=str, default=None, help="CAREFUL: old checkpoint is deleted (TODO)")
    parser = Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = LearnedEmbedding.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
