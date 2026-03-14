"""Multi-label classification metrics."""

import json
from typing import Dict, List

import jsonlines
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


def calculate_multilabel_metrics(
    predictions_jsonl: str,
    ground_truth_jsonl: str,
    pred_key: str = "video_level_tags",
    gt_key: str = "tags",
    exclude_tags: List[str] = None,
    include_tags: List[str] = None,
    exclude_failures: bool = True,
) -> Dict:
    """
    Calculate multi-label classification metrics comparing predictions to ground truth.

    Args:
        predictions_jsonl: Path to JSONL file with predictions (output of videos_to_jsonl_video_level)
        ground_truth_jsonl: Path to JSONL file with ground truth labels
        pred_key: Key name for predicted tags in predictions file
        gt_key: Key name for ground truth tags in ground truth file
        exclude_tags: List of tags to exclude from metric calculation (still predicted, but not evaluated)
        include_tags: List of tags to include in metric calculation (if specified, only these tags are evaluated)
        exclude_failures: If True, exclude videos with "error" field from metrics (default: True)

    Returns:
        Dictionary containing various multi-label classification metrics
    """
    exclude_tags = exclude_tags or []
    include_tags = include_tags or []

    # Load predictions, optionally excluding failures
    predictions = {}
    failed_count = 0
    with jsonlines.open(predictions_jsonl) as reader:
        for line in reader:
            video_id = line["video_id"]
            # Check if this is a failed video (has "error" field)
            if "error" in line:
                failed_count += 1
                if exclude_failures:
                    continue  # Skip failed videos
            predictions[video_id] = line.get(pred_key, [])

    # Load ground truth
    ground_truth = {}
    with jsonlines.open(ground_truth_jsonl) as reader:
        for line in reader:
            video_id = line["video_id"]
            ground_truth[video_id] = line.get(gt_key, [])

    # Find common video IDs
    common_ids = sorted(set(predictions.keys()) & set(ground_truth.keys()))

    if not common_ids:
        raise ValueError("No common video IDs found between predictions and ground truth")

    if failed_count > 0:
        if exclude_failures:
            logger.info(f"Excluded {failed_count} failed videos from evaluation")
        else:
            logger.info(f"Included {failed_count} failed videos (with empty predictions) in evaluation")

    logger.info(f"Evaluating {len(common_ids)} videos")

    # Align predictions and ground truth
    y_pred_tags = [predictions[vid] for vid in common_ids]
    y_true_tags = [ground_truth[vid] for vid in common_ids]

    # Filter out excluded tags from both predictions and ground truth
    if exclude_tags:
        logger.info(f"Excluding {len(exclude_tags)} tags from evaluation: {exclude_tags}")
        y_pred_tags = [[tag for tag in tags if tag not in exclude_tags] for tags in y_pred_tags]
        y_true_tags = [[tag for tag in tags if tag not in exclude_tags] for tags in y_true_tags]

    # Filter to only included tags (applied after exclude_tags)
    if include_tags:
        logger.info(f"Including only {len(include_tags)} tags in evaluation: {include_tags}")
        y_pred_tags = [[tag for tag in tags if tag in include_tags] for tags in y_pred_tags]
        y_true_tags = [[tag for tag in tags if tag in include_tags] for tags in y_true_tags]

    # Convert to binary label matrix using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit on both to ensure all labels are captured
    mlb.fit(y_true_tags + y_pred_tags)

    y_true = mlb.transform(y_true_tags)
    y_pred = mlb.transform(y_pred_tags)

    num_labels = len(mlb.classes_)
    logger.info(f"Using {num_labels} labels for evaluation")

    # Check if we have any labels after filtering
    if num_labels == 0:
        logger.error("No labels remaining after filtering! Cannot calculate metrics.")
        logger.error("This might happen if include_tags/test_tags don't match any ground truth or predictions.")
        raise ValueError(
            "No labels available for metric calculation after filtering. "
            "Check that your test_tags/include_tags exist in both predictions and ground truth."
        )

    # Calculate metrics
    # Note: "samples" averaging requires multilabel format (more than 1 label)
    # For single-label scenarios, samples metrics will be set to None
    is_multilabel = num_labels > 1

    metrics = {
        # Overall metrics
        "subset_accuracy": accuracy_score(y_true, y_pred),  # Exact match ratio
        "hamming_loss": hamming_loss(y_true, y_pred),
        # Aggregated metrics (micro, macro, weighted, samples)
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "jaccard_micro": jaccard_score(y_true, y_pred, average="micro", zero_division=0),
        "jaccard_macro": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
        "jaccard_weighted": jaccard_score(y_true, y_pred, average="weighted", zero_division=0),
        # Per-label metrics
        "labels": mlb.classes_.tolist(),
        "num_videos": len(common_ids),
        "common_video_ids": common_ids,
    }

    # Add "samples" averaging metrics only for multilabel scenarios (>1 label)
    if is_multilabel:
        metrics["precision_samples"] = precision_score(y_true, y_pred, average="samples", zero_division=0)
        metrics["recall_samples"] = recall_score(y_true, y_pred, average="samples", zero_division=0)
        metrics["f1_samples"] = f1_score(y_true, y_pred, average="samples", zero_division=0)
        metrics["jaccard_samples"] = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    else:
        # Set to None for single-label scenarios
        metrics["precision_samples"] = None
        metrics["recall_samples"] = None
        metrics["f1_samples"] = None
        metrics["jaccard_samples"] = None

    # Per-label precision, recall, F1
    (
        precision_per_label,
        recall_per_label,
        f1_per_label,
        support,
    ) = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    metrics["per_label_metrics"] = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for label, p, r, f, s in zip(mlb.classes_, precision_per_label, recall_per_label, f1_per_label, support)
    }

    return metrics


def print_metrics_summary(metrics: Dict) -> None:
    """Pretty print metrics summary."""
    print("=" * 60)
    print("MULTI-LABEL CLASSIFICATION METRICS")
    print("=" * 60)
    print(f"\nDataset: {metrics['num_videos']} videos")
    print(f"Labels: {len(metrics['labels'])} unique tags")
    print(f"  {metrics['labels']}")

    print("\n" + "-" * 60)
    print("OVERALL METRICS")
    print("-" * 60)
    print(f"Subset Accuracy (Exact Match): {metrics['subset_accuracy']:.4f}")
    print(f"Hamming Loss:                   {metrics['hamming_loss']:.4f}")

    print("\n" + "-" * 60)
    print("AGGREGATED METRICS")
    print("-" * 60)
    print(f"{'Metric':<20} {'Micro':<10} {'Macro':<10} {'Weighted':<10} {'Samples':<10}")
    print("-" * 60)

    # Helper function to format metric value (handle None for samples)
    def fmt_metric(value):
        return f"{value:<10.4f}" if value is not None else f"{'N/A':<10}"

    print(
        f"{'Precision':<20} {metrics['precision_micro']:<10.4f} {metrics['precision_macro']:<10.4f} "
        f"{metrics['precision_weighted']:<10.4f} {fmt_metric(metrics['precision_samples'])}"
    )
    print(
        f"{'Recall':<20} {metrics['recall_micro']:<10.4f} {metrics['recall_macro']:<10.4f} "
        f"{metrics['recall_weighted']:<10.4f} {fmt_metric(metrics['recall_samples'])}"
    )
    print(
        f"{'F1-Score':<20} {metrics['f1_micro']:<10.4f} {metrics['f1_macro']:<10.4f} "
        f"{metrics['f1_weighted']:<10.4f} {fmt_metric(metrics['f1_samples'])}"
    )
    print(
        f"{'Jaccard':<20} {metrics['jaccard_micro']:<10.4f} {metrics['jaccard_macro']:<10.4f} "
        f"{metrics['jaccard_weighted']:<10.4f} {fmt_metric(metrics['jaccard_samples'])}"
    )

    print("\n" + "-" * 60)
    print("PER-LABEL METRICS")
    print("-" * 60)
    print(f"{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for label in sorted(metrics["per_label_metrics"].keys()):
        lm = metrics["per_label_metrics"][label]
        print(f"{label:<15} {lm['precision']:<12.4f} {lm['recall']:<12.4f} {lm['f1']:<12.4f} {lm['support']:<10}")
    print("=" * 60)


def save_metrics_to_json(metrics: Dict, output_path: str) -> None:
    """Save metrics dictionary to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to: {output_path}")
