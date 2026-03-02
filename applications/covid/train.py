"""
COVID severity classification training entry point.

Uses TissueFormer with GroupedDonorTrainer for group_size > 1,
and standard HuggingFace Trainer with BertForSequenceClassification for group_size = 1.
"""

import os
import sys
import json
from typing import Dict, Optional

import hydra
import wandb
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk, Dataset, DatasetDict, disable_caching
from transformers import (
    TrainingArguments,
    Trainer,
    BertForSequenceClassification,
    set_seed,
)
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from tissueformer.model import TissueFormer, TissueFormerConfig
from tissueformer.class_weights import calculate_class_weights
from tissueformer.samplers import GroupedDonorTrainer
from transformers import BertModel


def setup_wandb(cfg: DictConfig):
    """Initialize W&B logging."""
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def create_model(config: DictConfig, class_weights=None):
    """Create model based on config."""
    class_weights_list = class_weights.tolist() if class_weights is not None else None

    bert_path = os.path.expanduser(config.model.bert_path_or_name)

    if config.model.pretrained_type == "single-cell":
        model = BertForSequenceClassification.from_pretrained(
            bert_path,
            num_labels=config.model.num_labels,
        )
    elif config.model.pretrained_type in ("none", "bert_only"):
        model_config = TissueFormerConfig(
            num_labels=config.model.num_labels,
            bert_config=bert_path,
            num_set_layers=config.model.num_set_layers,
            set_hidden_size=config.model.set_hidden_size,
            num_attention_heads=config.model.num_attention_heads,
            dropout_prob=config.model.dropout_prob,
            class_weights=class_weights_list,
            single_cell_vs_group_weight=config.model.single_cell_vs_group_weight,
            use_relative_positions=False,
            rms_layernorm=config.model.get("rms_layernorm", False),
        )
        model = TissueFormer(model_config)

        if config.model.pretrained_type == "bert_only":
            pretrained_bert = BertModel.from_pretrained(bert_path)
            model.bert.load_state_dict(pretrained_bert.state_dict())

        if hasattr(model, "class_weights") and class_weights is not None:
            model.class_weights = class_weights
    else:
        raise ValueError(f"Unknown pretrained_type: {config.model.pretrained_type}")

    return model


def get_class_weights(cfg: DictConfig, train_labels: np.ndarray):
    """Compute or load class weights for imbalanced classification.

    When path is provided, loads pre-computed weights from file.
    Otherwise, computes weights from train_labels using the configured method.
    """
    if not cfg.data.class_weights.enabled:
        return None

    if cfg.data.class_weights.path is not None:
        weights = np.load(cfg.data.class_weights.path)
    else:
        method = cfg.data.class_weights.get("method", "balanced")
        weights = calculate_class_weights(train_labels, method=method)

    if len(weights) != cfg.model.num_labels:
        raise ValueError(
            f"Number of class weights ({len(weights)}) != num_labels ({cfg.model.num_labels})"
        )
    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights ({cfg.data.class_weights.get('method', 'loaded')}): {weights.tolist()}")
    wandb.run.summary["class_weights"] = weights.tolist()
    return weights


def prepare_datasets(dataset, donor_splits, cv_fold, config):
    """Prepare train/validation/test splits from a single tokenized dataset.

    The full tokenized dataset is split into train/test using donor
    assignments from *donor_splits*, then train is further divided into
    train/validation.
    """
    fold_info = donor_splits["folds"][str(cv_fold)]
    train_donors = set(fold_info["train_donors"])
    test_donors = set(fold_info["test_donors"])

    donor_ids = np.array(dataset["donor_id"])
    train_mask = np.isin(donor_ids, list(train_donors))
    test_mask = np.isin(donor_ids, list(test_donors))

    train_dataset = dataset.select(np.where(train_mask)[0])
    test_dataset = dataset.select(np.where(test_mask)[0])

    train_idx, val_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=config.data.validation_split,
        random_state=config.seed,
    )

    # Add unique IDs
    test_dataset = test_dataset.add_column(
        "uuid", np.arange(len(test_dataset))
    )
    train_dataset = train_dataset.add_column(
        "uuid", np.arange(len(train_dataset))
    )

    val_dataset = train_dataset.select(val_idx)
    train_dataset = train_dataset.select(train_idx)

    if hasattr(config.data, "max_train_samples") and config.data.max_train_samples:
        train_dataset = train_dataset.select(
            range(min(len(train_dataset), config.data.max_train_samples))
        )
    if hasattr(config.data, "max_eval_samples") and config.data.max_eval_samples:
        val_dataset = val_dataset.select(
            range(min(len(val_dataset), config.data.max_eval_samples))
        )
        test_dataset = test_dataset.select(
            range(min(len(test_dataset), config.data.max_eval_samples))
        )

    # Rename label column to 'labels' for HF Trainer
    if hasattr(config.data, "label_key"):
        train_dataset = train_dataset.rename_column(config.data.label_key, "labels")
        val_dataset = val_dataset.rename_column(config.data.label_key, "labels")
        test_dataset = test_dataset.rename_column(config.data.label_key, "labels")

    # Verify label range
    all_labels = np.array(train_dataset["labels"])
    unique_labels = np.unique(all_labels)
    if unique_labels.min() < 0 or unique_labels.max() >= config.model.num_labels:
        raise ValueError(
            f"Labels must be in [0, {config.model.num_labels - 1}], "
            f"found range [{unique_labels.min()}, {unique_labels.max()}]"
        )

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })


def compute_metrics(eval_pred, label_names=None, num_labels=3):
    """Compute accuracy, F1, and AUROC."""
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    if label_names:
        id2label = {int(k): v for k, v in label_names.items()}
    else:
        logits_shape = logits[0].shape[-1] if isinstance(logits, tuple) else logits.shape[-1]
        id2label = {i: str(i) for i in range(logits_shape)}

    all_label_ids = list(range(logits[0].shape[-1] if isinstance(logits, tuple) else logits.shape[-1]))
    target_names = [id2label[i] for i in all_label_ids]

    if isinstance(labels, tuple):
        labels = labels[0]
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    report = classification_report(
        labels, predictions,
        output_dict=True,
        labels=all_label_ids,
        target_names=target_names,
        zero_division=0,
    )

    metrics = {
        "accuracy": (predictions == labels).mean(),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }

    # Compute AUROC (handle edge cases)
    try:
        from scipy.special import softmax
        probs = softmax(logits, axis=-1)
        if len(all_label_ids) == 2:
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        metrics["auroc"] = auroc
    except ValueError:
        # Can happen if a class is missing from labels in this eval batch
        pass

    return metrics


def aggregate_donor_predictions(predictions, logits, donor_ids, labels):
    """
    Aggregate single-cell predictions to donor level by majority vote.
    Also aggregates logits by mean for AUROC computation.
    """
    unique_donors = np.unique(donor_ids)
    donor_preds = []
    donor_logits = []
    donor_labels = []

    for donor in unique_donors:
        mask = donor_ids == donor
        # Majority vote
        pred_counts = np.bincount(predictions[mask])
        donor_preds.append(pred_counts.argmax())
        # Mean logits for AUROC
        donor_logits.append(logits[mask].mean(axis=0))
        # All cells from a donor should have the same label
        donor_labels.append(labels[mask][0])

    return (
        np.array(donor_preds),
        np.array(donor_logits),
        np.array(donor_labels),
        unique_donors,
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    use_single_cell = cfg.model.pretrained_type == "single-cell" or cfg.data.group_size == 1
    if not use_single_cell:
        assert cfg.data.group_size is not None, "group_size required for grouped models"
        assert not cfg.training.remove_unused_columns, \
            "remove_unused_columns must be False for grouped models"

    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)
        cfg.data.max_train_samples = 200
        cfg.data.max_eval_samples = 200
        cfg.training.report_to = None

    set_seed(cfg.seed)
    setup_wandb(cfg)
    disable_caching()

    # Load dataset and donor splits
    dataset = load_from_disk(cfg.data.dataset_path)
    if isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected a single Dataset at {cfg.data.dataset_path}, "
            "got a DatasetDict. Re-run tokenization with the updated "
            "tokenize_cells.py (tokenize_dataset) to produce a single Dataset."
        )

    with open(cfg.data.splits_path) as f:
        donor_splits = json.load(f)

    datasets = prepare_datasets(dataset, donor_splits, cfg.data.cv_fold, cfg)
    print(f"Loaded datasets: {datasets}")

    # Compute or load class weights from training labels
    train_labels = np.array(datasets["train"]["labels"])
    class_weights = get_class_weights(cfg, train_labels)
    model = create_model(cfg, class_weights)

    trainer_params = {
        "model": model,
        "args": TrainingArguments(**cfg.training),
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "compute_metrics": lambda pred: compute_metrics(
            pred, cfg.data.label_names, cfg.model.num_labels
        ),
    }

    if use_single_cell:
        trainer = Trainer(**trainer_params)
    else:
        trainer = GroupedDonorTrainer(
            **trainer_params,
            donor_group_size=cfg.data.group_size,
            donor_key=cfg.data.donor_key,
            label_key="labels",
            index_key="uuid",
        )

    # Train
    if cfg.training.num_train_epochs > 0:
        train_result = trainer.train()
        trainer.save_model()
        trainer.save_state()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Soft check: warn if loss didn't decrease
        if "train_loss" in metrics:
            print(f"Final training loss: {metrics['train_loss']:.4f}")

        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    # Test
    if cfg.run_test_set:
        for data_key in ["test", "validation"]:
            trainer.accelerator.gradient_state._reset_state()

            if use_single_cell:
                outputs = trainer.predict(datasets[data_key], metric_key_prefix=data_key)
                logits = outputs.predictions
                predictions = np.argmax(logits, axis=-1)
                labels = outputs.label_ids

                # Aggregate to donor level
                donor_ids = np.array(datasets[data_key]["donor_id"])
                donor_preds, donor_logits, donor_labels, donors = \
                    aggregate_donor_predictions(predictions, logits, donor_ids, labels)

                output_dict = {
                    "cell_predictions": predictions,
                    "cell_labels": labels,
                    "donor_predictions": donor_preds,
                    "donor_logits": donor_logits,
                    "donor_labels": donor_labels,
                    "donor_ids": donors,
                    "label_names": dict(cfg.data.label_names),
                }
            else:
                outputs, indices = trainer.predict(
                    datasets[data_key], metric_key_prefix=data_key
                )
                if isinstance(outputs.predictions, tuple):
                    predictions = np.argmax(outputs.predictions[0], axis=-1)
                else:
                    predictions = np.argmax(outputs.predictions, axis=-1)
                labels = outputs.label_ids[0] if isinstance(outputs.label_ids, tuple) else outputs.label_ids

                output_dict = {
                    "predictions": predictions,
                    "labels": labels,
                    "indices": indices,
                    "label_names": dict(cfg.data.label_names),
                }

            trainer.log_metrics(data_key, outputs.metrics)
            np.save(
                os.path.join(cfg.training.output_dir, f"{data_key}_predictions.npy"),
                output_dict,
            )

    wandb.finish()


if __name__ == "__main__":
    main()
