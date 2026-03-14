"""Tests for multi-label classification metrics calculation."""

import pytest

from benchmarks.metrics.multilabel_classification import MultilabelClassificationMetric
from tests.fixtures.classification_data import create_ground_truth, create_predictions, write_to_jsonl


@pytest.mark.fast
def test_perfect_predictions(tmp_path):
    """Test metrics when predictions exactly match ground truth."""
    video_ids = ["vid_1", "vid_2", "vid_3"]
    tags_per_video = {
        "vid_1": ["chocolate", "cheerful"],
        "vid_2": ["cars"],
        "vid_3": ["chocolate", "cars", "cheerful"],
    }

    predictions = create_predictions(video_ids, tags_per_video)
    ground_truth = create_ground_truth(video_ids, tags_per_video)

    pred_file = tmp_path / "predictions.jsonl"
    gt_file = tmp_path / "ground_truth.jsonl"
    write_to_jsonl(predictions, pred_file)
    write_to_jsonl(ground_truth, gt_file)

    calculator = MultilabelClassificationMetric()
    metrics = calculator.calculate(predictions_path=str(pred_file), ground_truth_path=str(gt_file))

    assert metrics["subset_accuracy"] == 1.0
    assert metrics["precision_micro"] == 1.0
    assert metrics["recall_micro"] == 1.0
    assert metrics["f1_micro"] == 1.0
    assert metrics["hamming_loss"] == 0.0


@pytest.mark.fast
def test_specific_per_label_metrics(tmp_path):
    """
    Test exact per-label metric values with hand-calculated expectations.

    Dataset (4 videos, 5 tags):
        vid_1: pred=[chocolate, cheerful], gt=[chocolate, cheerful]
        vid_2: pred=[cars, food], gt=[cars]
        vid_3: pred=[chocolate], gt=[chocolate, tech]
        vid_4: pred=[tech], gt=[food, tech]

    Hand-calculated expectations:

    chocolate: TP=2 (vid_1, vid_3), FP=0, FN=0, support=2
        precision=1.0, recall=1.0, f1=1.0

    cheerful: TP=1 (vid_1), FP=0, FN=0, support=1
        precision=1.0, recall=1.0, f1=1.0

    cars: TP=1 (vid_2), FP=0, FN=0, support=1
        precision=1.0, recall=1.0, f1=1.0

    food: TP=0, FP=1 (vid_2), FN=1 (vid_4), support=1
        precision=0.0, recall=0.0, f1=0.0

    tech: TP=1 (vid_4), FP=0, FN=1 (vid_3), support=2
        precision=1.0, recall=0.5, f1=0.6667
    """
    predictions_tags = {
        "vid_1": ["chocolate", "cheerful"],
        "vid_2": ["cars", "food"],
        "vid_3": ["chocolate"],
        "vid_4": ["tech"],
    }

    gt_tags = {
        "vid_1": ["chocolate", "cheerful"],
        "vid_2": ["cars"],
        "vid_3": ["chocolate", "tech"],
        "vid_4": ["food", "tech"],
    }

    video_ids = list(predictions_tags.keys())
    predictions = create_predictions(video_ids, predictions_tags)
    ground_truth = create_ground_truth(video_ids, gt_tags)

    pred_file = tmp_path / "predictions.jsonl"
    gt_file = tmp_path / "ground_truth.jsonl"
    write_to_jsonl(predictions, pred_file)
    write_to_jsonl(ground_truth, gt_file)

    calculator = MultilabelClassificationMetric()
    metrics = calculator.calculate(predictions_path=str(pred_file), ground_truth_path=str(gt_file))

    per_label = metrics["per_label_metrics"]

    # chocolate: perfect (TP=2, FP=0, FN=0)
    assert per_label["chocolate"]["precision"] == 1.0
    assert per_label["chocolate"]["recall"] == 1.0
    assert per_label["chocolate"]["f1"] == 1.0
    assert per_label["chocolate"]["support"] == 2

    # cheerful: perfect (TP=1, FP=0, FN=0)
    assert per_label["cheerful"]["precision"] == 1.0
    assert per_label["cheerful"]["recall"] == 1.0
    assert per_label["cheerful"]["f1"] == 1.0
    assert per_label["cheerful"]["support"] == 1

    # cars: perfect (TP=1, FP=0, FN=0)
    assert per_label["cars"]["precision"] == 1.0
    assert per_label["cars"]["recall"] == 1.0
    assert per_label["cars"]["f1"] == 1.0
    assert per_label["cars"]["support"] == 1

    # food: all wrong (TP=0, FP=1, FN=1)
    assert per_label["food"]["precision"] == 0.0
    assert per_label["food"]["recall"] == 0.0
    assert per_label["food"]["f1"] == 0.0
    assert per_label["food"]["support"] == 1

    # tech: partial (TP=1, FP=0, FN=1)
    assert per_label["tech"]["precision"] == 1.0
    assert per_label["tech"]["recall"] == 0.5
    assert abs(per_label["tech"]["f1"] - 0.6667) < 0.001
    assert per_label["tech"]["support"] == 2


@pytest.mark.fast
def test_no_predictions(tmp_path):
    """Test metrics when all predictions are empty."""
    video_ids = ["vid_1", "vid_2", "vid_3"]
    predictions_tags = {
        "vid_1": [],
        "vid_2": [],
        "vid_3": [],
    }
    gt_tags = {
        "vid_1": ["chocolate", "cheerful"],
        "vid_2": ["cars"],
        "vid_3": ["chocolate"],
    }

    predictions = create_predictions(video_ids, predictions_tags)
    ground_truth = create_ground_truth(video_ids, gt_tags)

    pred_file = tmp_path / "predictions.jsonl"
    gt_file = tmp_path / "ground_truth.jsonl"
    write_to_jsonl(predictions, pred_file)
    write_to_jsonl(ground_truth, gt_file)

    calculator = MultilabelClassificationMetric()
    metrics = calculator.calculate(predictions_path=str(pred_file), ground_truth_path=str(gt_file))

    assert metrics["precision_micro"] == 0.0
    assert metrics["recall_micro"] == 0.0
    assert metrics["f1_micro"] == 0.0
    assert metrics["hamming_loss"] > 0


@pytest.mark.fast
def test_exclude_failed_videos(tmp_path):
    """Test that videos with 'error' field are excluded when exclude_failures=True."""
    video_ids = ["vid_1", "vid_2", "vid_3"]
    predictions_tags = {"vid_1": ["chocolate"], "vid_2": ["cars"]}
    gt_tags = {"vid_1": ["chocolate"], "vid_2": ["cars"], "vid_3": ["cheerful"]}

    predictions = create_predictions(video_ids, predictions_tags, include_failures=["vid_3"])
    ground_truth = create_ground_truth(video_ids, gt_tags)

    pred_file = tmp_path / "predictions.jsonl"
    gt_file = tmp_path / "ground_truth.jsonl"
    write_to_jsonl(predictions, pred_file)
    write_to_jsonl(ground_truth, gt_file)

    calculator = MultilabelClassificationMetric(exclude_failures=True)
    metrics = calculator.calculate(predictions_path=str(pred_file), ground_truth_path=str(gt_file))

    assert metrics["num_videos"] == 2


@pytest.mark.fast
def test_include_failed_videos(tmp_path):
    """Test that videos with 'error' field are included as empty predictions when exclude_failures=False."""
    video_ids = ["vid_1", "vid_2", "vid_3"]
    predictions_tags = {"vid_1": ["chocolate"], "vid_2": ["cars"]}
    gt_tags = {"vid_1": ["chocolate"], "vid_2": ["cars"], "vid_3": ["cheerful"]}

    predictions = create_predictions(video_ids, predictions_tags, include_failures=["vid_3"])
    ground_truth = create_ground_truth(video_ids, gt_tags)

    pred_file = tmp_path / "predictions.jsonl"
    gt_file = tmp_path / "ground_truth.jsonl"
    write_to_jsonl(predictions, pred_file)
    write_to_jsonl(ground_truth, gt_file)

    calculator = MultilabelClassificationMetric(exclude_failures=False)
    metrics = calculator.calculate(predictions_path=str(pred_file), ground_truth_path=str(gt_file))

    assert metrics["num_videos"] == 3


@pytest.mark.fast
def test_exclude_tags(tmp_path):
    """Test that excluded tags are removed from both predictions and ground truth."""
    video_ids = ["vid_1", "vid_2"]
    predictions_tags = {
        "vid_1": ["chocolate", "cheerful", "excluded_tag"],
        "vid_2": ["cars", "excluded_tag"],
    }
    gt_tags = {
        "vid_1": ["chocolate", "cheerful"],
        "vid_2": ["cars", "excluded_tag"],
    }

    predictions = create_predictions(video_ids, predictions_tags)
    ground_truth = create_ground_truth(video_ids, gt_tags)

    pred_file = tmp_path / "predictions.jsonl"
    gt_file = tmp_path / "ground_truth.jsonl"
    write_to_jsonl(predictions, pred_file)
    write_to_jsonl(ground_truth, gt_file)

    calculator = MultilabelClassificationMetric(exclude_tags=["excluded_tag"])
    metrics = calculator.calculate(predictions_path=str(pred_file), ground_truth_path=str(gt_file))

    assert "excluded_tag" not in metrics["labels"]
    assert len(metrics["labels"]) == 3


@pytest.mark.fast
def test_include_tags(tmp_path):
    """Test that only included tags are evaluated."""
    video_ids = ["vid_1", "vid_2"]
    predictions_tags = {
        "vid_1": ["chocolate", "cheerful", "cars"],
        "vid_2": ["cars", "other_tag"],
    }
    gt_tags = {
        "vid_1": ["chocolate", "cheerful"],
        "vid_2": ["cars", "other_tag"],
    }

    predictions = create_predictions(video_ids, predictions_tags)
    ground_truth = create_ground_truth(video_ids, gt_tags)

    pred_file = tmp_path / "predictions.jsonl"
    gt_file = tmp_path / "ground_truth.jsonl"
    write_to_jsonl(predictions, pred_file)
    write_to_jsonl(ground_truth, gt_file)

    calculator = MultilabelClassificationMetric(include_tags=["chocolate", "cars"])
    metrics = calculator.calculate(predictions_path=str(pred_file), ground_truth_path=str(gt_file))

    assert set(metrics["labels"]) == {"chocolate", "cars"}
    assert "cheerful" not in metrics["labels"]
    assert "other_tag" not in metrics["labels"]


@pytest.mark.fast
def test_no_common_video_ids(tmp_path):
    """Test that ValueError is raised when no common video IDs exist."""
    predictions = create_predictions(["vid_1", "vid_2"], {"vid_1": ["chocolate"], "vid_2": ["cars"]})
    ground_truth = create_ground_truth(["vid_3", "vid_4"], {"vid_3": ["cheerful"], "vid_4": ["cars"]})

    pred_file = tmp_path / "predictions.jsonl"
    gt_file = tmp_path / "ground_truth.jsonl"
    write_to_jsonl(predictions, pred_file)
    write_to_jsonl(ground_truth, gt_file)

    calculator = MultilabelClassificationMetric()

    with pytest.raises(ValueError, match="No common video IDs"):
        calculator.calculate(predictions_path=str(pred_file), ground_truth_path=str(gt_file))
