"""
COVID severity classification training entry point.

Uses TissueFormer with GroupedDonorTrainer for group_size > 1,
and standard HuggingFace Trainer with BertForSequenceClassification for group_size = 1.
"""

import os
import sys
import json
import glob
import signal
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
    EarlyStoppingCallback,
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
            bert_micro_batch_size=config.model.get("bert_micro_batch_size", 0),
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
    train/validation by holding out entire donors (not random cells).
    """
    fold_info = donor_splits["folds"][str(cv_fold)]
    train_donors_list = list(fold_info["train_donors"])
    test_donors = set(fold_info["test_donors"])

    donor_ids = np.array(dataset["donor_id"])

    # Split train donors into actual_train and val donors (donor-level)
    # Get one label per donor for stratification
    donor_labels = {}
    label_key = config.data.get("label_key", "label")
    for d in train_donors_list:
        mask = donor_ids == d
        donor_labels[d] = dataset[label_key][np.where(mask)[0][0]]

    n_val = max(1, round(len(train_donors_list) * config.data.validation_split))
    if n_val >= len(train_donors_list):
        n_val = max(1, len(train_donors_list) - 1)

    donor_label_arr = np.array([donor_labels[d] for d in train_donors_list])
    actual_train_donors, val_donors = train_test_split(
        train_donors_list,
        test_size=n_val,
        random_state=config.seed,
        stratify=donor_label_arr,
    )

    actual_train_donors = set(actual_train_donors)
    val_donors = set(val_donors)
    print(f"  Donor split: {len(actual_train_donors)} train, {len(val_donors)} val, {len(test_donors)} test")
    print(f"  Val donors: {sorted(val_donors)}")

    train_mask = np.isin(donor_ids, list(actual_train_donors))
    val_mask = np.isin(donor_ids, list(val_donors))
    test_mask = np.isin(donor_ids, list(test_donors))

    train_dataset = dataset.select(np.where(train_mask)[0])
    val_dataset = dataset.select(np.where(val_mask)[0])
    test_dataset = dataset.select(np.where(test_mask)[0])

    # Add unique IDs
    test_dataset = test_dataset.add_column(
        "uuid", np.arange(len(test_dataset))
    )
    train_dataset = train_dataset.add_column(
        "uuid", np.arange(len(train_dataset))
    )
    val_dataset = val_dataset.add_column(
        "uuid", np.arange(len(val_dataset))
    )

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


def aggregate_to_donor(logits, predictions, donor_ids, labels):
    """Aggregate group/cell predictions to donor level using two methods.

    Args:
        logits: (n_groups, n_classes) raw model logits per group/cell
        predictions: (n_groups,) argmax predictions per group/cell
        donor_ids: (n_groups,) donor ID for each group/cell
        labels: (n_groups,) label for each group/cell

    Returns:
        dict with keys 'majority_vote' and 'mean_logits', each containing
        'predictions', 'labels', and 'donor_ids' arrays. 'mean_logits' also
        contains 'logits' for AUROC computation.
    """
    unique_donors = np.unique(donor_ids)
    mv_preds = []
    ml_logits = []
    donor_labels = []

    for donor in unique_donors:
        mask = donor_ids == donor
        # Majority vote
        pred_counts = np.bincount(predictions[mask])
        mv_preds.append(pred_counts.argmax())
        # Mean logits
        ml_logits.append(logits[mask].mean(axis=0))
        # All groups from a donor share the same label
        donor_labels.append(labels[mask][0])

    mv_preds = np.array(mv_preds)
    ml_logits = np.array(ml_logits)
    ml_preds = np.argmax(ml_logits, axis=-1)
    donor_labels = np.array(donor_labels)

    return {
        "majority_vote": {
            "predictions": mv_preds,
            "labels": donor_labels,
            "donor_ids": unique_donors,
        },
        "mean_logits": {
            "predictions": ml_preds,
            "logits": ml_logits,
            "labels": donor_labels,
            "donor_ids": unique_donors,
        },
    }


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

    callbacks = []
    if cfg.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience,
        ))

    trainer_params = {
        "model": model,
        "args": TrainingArguments(**cfg.training),
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "compute_metrics": lambda pred: compute_metrics(
            pred, cfg.data.label_names, cfg.model.num_labels
        ),
        "callbacks": callbacks,
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

    # Resume from checkpoint if a previous run was preempted
    breadcrumb_dir = os.path.join("outputs", cfg.dataset_name)
    breadcrumb_path = os.path.join(
        breadcrumb_dir,
        f".resume_fold{cfg.data.cv_fold}_gs{cfg.data.group_size}.json",
    )
    resume_from = None
    if os.path.exists(breadcrumb_path):
        with open(breadcrumb_path) as f:
            prev_output_dir = json.load(f)["output_dir"]
        checkpoints = sorted(glob.glob(os.path.join(prev_output_dir, "checkpoint-*")))
        if checkpoints:
            resume_from = checkpoints[-1]
            print(f"Resuming from checkpoint: {resume_from}")

    # Write breadcrumb so a requeued job can find our checkpoints
    os.makedirs(breadcrumb_dir, exist_ok=True)
    with open(breadcrumb_path, "w") as f:
        json.dump({"output_dir": cfg.training.output_dir}, f)

    # Handle SIGUSR1 from SLURM preemption: save checkpoint and exit
    def handle_preempt(signum, frame):
        print("SIGUSR1 received — saving checkpoint before preemption")
        trainer.save_model(os.path.join(cfg.training.output_dir, "checkpoint-preempt"))
        trainer.save_state()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, handle_preempt)

    # Train
    if cfg.training.num_train_epochs > 0:
        train_result = trainer.train(resume_from_checkpoint=resume_from)
        trainer.save_model()
        trainer.save_state()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Soft check: warn if loss didn't decrease
        if "train_loss" in metrics:
            print(f"Final training loss: {metrics['train_loss']:.4f}")

    # Shared helper: predict + compute group/donor metrics + log to wandb
    def predict_and_log(data_key):
        from transformers.trainer_utils import EvalPrediction

        trainer.accelerator.gradient_state._reset_state()

        if use_single_cell:
            outputs = trainer.predict(datasets[data_key], metric_key_prefix=data_key)
            logits = outputs.predictions
            predictions = np.argmax(logits, axis=-1)
            labels = outputs.label_ids
            donor_ids = np.array(datasets[data_key]["donor_id"])
        else:
            outputs, indices, donor_ids = trainer.predict(
                datasets[data_key], metric_key_prefix=data_key
            )
            logits = outputs.predictions[0] if isinstance(outputs.predictions, tuple) else outputs.predictions
            predictions = np.argmax(logits, axis=-1)
            labels = outputs.label_ids[0] if isinstance(outputs.label_ids, tuple) else outputs.label_ids

        # Group-level metrics (cell-level for single-cell mode)
        group_metrics = compute_metrics(
            EvalPrediction(predictions=logits, label_ids=labels),
            cfg.data.label_names, cfg.model.num_labels,
        )
        group_metrics = {f"group_{k}": v for k, v in group_metrics.items()}

        # Aggregate to donor level (both methods)
        donor_agg = aggregate_to_donor(logits, predictions, donor_ids, labels)

        # Donor metrics via majority vote (no AUROC — hard predictions only)
        mv = donor_agg["majority_vote"]
        mv_metrics = {
            "donor_majority_accuracy": (mv["predictions"] == mv["labels"]).mean(),
            "donor_majority_balanced_accuracy": balanced_accuracy_score(
                mv["labels"], mv["predictions"]
            ),
        }

        # Donor metrics via mean logits (full metrics including AUROC)
        ml = donor_agg["mean_logits"]
        ml_metrics = compute_metrics(
            EvalPrediction(predictions=ml["logits"], label_ids=ml["labels"]),
            cfg.data.label_names, cfg.model.num_labels,
        )
        ml_metrics = {f"donor_meanlogits_{k}": v for k, v in ml_metrics.items()}

        all_metrics = {**outputs.metrics, **group_metrics, **mv_metrics, **ml_metrics}
        trainer.log_metrics(data_key, all_metrics)

        # Log to wandb
        wandb_metrics = {f"{data_key}/{k}": v for k, v in all_metrics.items()}
        wandb_metrics[f"{data_key}/n_donors"] = len(mv["donor_ids"])
        wandb_metrics[f"{data_key}/n_groups"] = len(predictions)
        wandb.log(wandb_metrics)

        return outputs, logits, predictions, labels, donor_ids, mv, ml, group_metrics, mv_metrics, ml_metrics

    # Eval with group/donor metrics
    if cfg.training.num_train_epochs > 0 and not cfg.run_test_set:
        predict_and_log("train")
        predict_and_log("validation")

    # Test
    if cfg.run_test_set:
        predict_and_log("train")
        for data_key in ["test", "validation"]:
            _, logits, predictions, labels, donor_ids, mv, ml, group_metrics, mv_metrics, ml_metrics = predict_and_log(data_key)

            output_dict = {
                "group_predictions": predictions,
                "group_logits": logits,
                "group_labels": labels,
                "group_donor_ids": donor_ids,
                "group_metrics": group_metrics,
                "donor_majority_predictions": mv["predictions"],
                "donor_majority_metrics": mv_metrics,
                "donor_meanlogits_predictions": ml["predictions"],
                "donor_meanlogits_logits": ml["logits"],
                "donor_meanlogits_metrics": ml_metrics,
                "donor_labels": mv["labels"],
                "donor_ids": mv["donor_ids"],
                "label_names": dict(cfg.data.label_names),
            }
            np.save(
                os.path.join(cfg.training.output_dir, f"{data_key}_predictions.npy"),
                output_dict,
            )

            # Save JSON results for plotting (compatible with benchmark format)
            gs_str = str(cfg.data.group_size)
            json_result = {
                "method": "tissueformer",
                "group_size": gs_str,
                "group_metrics": {k: float(v) for k, v in group_metrics.items()},
                "donor_majority_metrics": {k: float(v) for k, v in mv_metrics.items()},
                "donor_meanlogits_metrics": {k: float(v) for k, v in ml_metrics.items()},
                "label_names": dict(cfg.data.label_names),
            }
            json_path = os.path.join(
                cfg.training.output_dir,
                f"tissueformer_gs{gs_str}_{data_key}_results.json",
            )
            with open(json_path, "w") as f:
                json.dump(json_result, f, indent=2)

    # Clean up breadcrumb on successful completion
    if os.path.exists(breadcrumb_path):
        os.remove(breadcrumb_path)

    wandb.finish()


if __name__ == "__main__":
    main()
