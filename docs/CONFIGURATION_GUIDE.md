# Configuration Guide

**DISCLAIMER**: The actual config files in `configs/` are hopefully self-explanatory and easy to follow. You can start using them directly without reading this entire guide as well. This reference document is here if you need clarification on specific parameters or want to understand what options are available. With that said:

This is a complete reference for the YAML configuration structure used across all tasks in the benchmarks framework. For model-specific parameters and requirements, see the [Model Reference Guide](MODEL_REFERENCE.md).

---

## Overview

**Config Directory Structure**:
```
configs/
├── standard_tagging/     # Multi-label classification with ground truth
├── workload/             # Cost/workload benchmarking
├── summarization/        # Video summary generation
├── summary_evaluation/   # LLM-as-a-judge evaluation
└── preprocessing/        # Standalone preprocessing scripts
```

**How Configs Work**:
1. YAML file defines task type, data sources, and pipeline components
2. Main entry point (`main.py`) routes to appropriate task module
3. Components are built dynamically from config using factory pattern
4. Pipeline executes with configured parameters

---

## Shared Configuration Sections

These sections are common across all tasks and models. Keys are consistent; values change based on task and model choice.

### Task Type

Routes to appropriate task module.

```yaml
task_type: "standard_tagging"
```

**Values**: `standard_tagging`, `cost_benchmark`, `summarization`, `summary_evaluation`

---

### Logging

Console output verbosity.

```yaml
logging:
  level: "INFO"
```

**Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`

---

### Tracking

Token usage and timing metrics.

```yaml
tracking:
  return_usage: true
  progress_interval: 50
```

- `return_usage`: Enable token/cost tracking (required for workload tasks)
- `progress_interval`: Log progress every N videos

---

### Pricing

Cost calculation (requires `return_usage: true`).

```yaml
pricing:
  input_per_1m_tokens: 0.33
  output_per_1m_tokens: 2.75
```

**Values**: Model-specific. See [Model Reference](MODEL_REFERENCE.md) for pricing.

---

### Video Source

Where videos are stored and how to access them.

```yaml
video_source:
  type: "s3"
  mode: "from_s3"
  s3:
    bucket: "your-s3-bucket"
    prefix: "videos"
    region: "us-east-2"
```

- `type`: Storage backend (`s3`, `gcs`, `local`)
- `mode`: How to provide videos to model (`from_s3`, `from_gcs`, `local`)

**Type + Mode Combinations**:

| type | mode | Use Case |
|------|------|----------|
| `s3` | `from_s3` | Native video API (Bedrock Nova, Pegasus) |
| `s3` | `local` | Frame-based models (GPT, NVIDIA, Qwen) |
| `gcs` | `from_gcs` | Native video API (Vertex AI Gemini) |
| `local` | `local` | Local testing |

---

### Video Selection

Which videos to process.

```yaml
video_selection:
  mode: "fixed"
  fixed_videos:
    - "vid_ABC.mp4"
    - "vid_DEF.mp4"
    ...
```

**Modes**:
- `fixed`: Specific video IDs (requires `fixed_videos` list)
- `random`: Random sample (requires `random_count`, optional `random_seed`)
- `all`: All videos from source
- `duration_based`: Target total duration
  - `target_minutes`: Total duration to target (e.g. 1000 in the workload pipeline)
  - `metadata_file`: Path to video metadata JSON file
  - `order`: Video ordering (`longest` or `shortest`)
  - `resolution`: Resolution filter (e.g., `1920x1080`) or `null` for all

---

### Paths

Input/output directories.

```yaml
paths:
  inputs_dir: "data/inputs"
  outputs_dir: "data/outputs"
  cache_dir: "data/cache/videos"
```

- `inputs_dir`: Pipeline inputs (ground truth labels, tag definitions, metadata)
- `outputs_dir`: Artifacts (logs, results JSONLs, metrics)
- `cache_dir`: Cached videos and/or frames

---

### Result Saver

Output file formatting.

```yaml
result_saver:
  type: "json"
  config:
    include_timestamp: true
```

- `include_timestamp`: Add timestamp to filenames (e.g., `predictions_model_20251215_143022.jsonl`)

---

## Task-Specific Configuration Sections

These sections vary by task type.

### Inference (Tagging Tasks)

Filters **predicted tags** based on model confidence scores.

```yaml
inference:
  threshold: 0.5
  ignore_threshold: true
  test_tags: null
```

- `threshold`: Minimum confidence score for predicted tags (0-1)
- `ignore_threshold`: If `true`, keep all predicted tags regardless of confidence
- `test_tags`: Optional list to limit evaluation to specific tags

---

### Labels (Tagging Tasks)

Filters **ground truth labels** based on annotation confidence scores.

```yaml
labels:
  gt_threshold: 0.5
```

- `gt_threshold`: Minimum confidence score for ground truth labels (0-1)

**Canonical Format**: Tagging tasks use a canonical JSON format for annotations and tag descriptions:

```yaml
paths:
  inputs_dir: "data/inputs"
  label_file: "video_annotations_youtube_ads.json"  # or "video_annotations_coactive.json"
  tag_descriptions_file: "tag_descriptions_youtube_ads.json"  # or "tag_descriptions_coactive.json"
  outputs_dir: "data/outputs"
  cache_dir: "data/cache/videos"
```

- `label_file`: Video annotations in canonical format (`{"video_id": ["tag1", "tag2"], ...}`)
- `tag_descriptions_file`: Tag descriptions in canonical format (`{"tag": "description", ...}`)

**Custom Prompts**: Optionally specify custom prompt files for different tag systems:

```yaml
pipeline:
  tagger:
    type: "bedrock_vision"
    config:
      custom_system_prompt: null  # Optional: custom system prompt filename
      custom_user_prompt: null    # Optional: custom user prompt filename
```

**Available Datasets**:
- **YouTube Ads (68 tags)**: `video_annotations_youtube_ads.json`, `tag_descriptions_youtube_ads.json`
  - Uses default model-specific prompts (e.g., `bedrock_user.txt`)
- **Coactive 100-tag**: `video_annotations_coactive.json`, `tag_descriptions_coactive.json`
  - **Important**: Set `custom_user_prompt: "coactive_user.txt"` when using this dataset
  - The Coactive prompt includes definitions for the 5 tag categories (Genre, Format, Subject, Mood, Theme) which are essential for correct tagging

**Example - Coactive 100-tag configuration**:
```yaml
paths:
  label_file: "video_annotations_coactive.json"
  tag_descriptions_file: "tag_descriptions_coactive.json"

pipeline:
  tagger:
    type: "bedrock_vision"
    config:
      model_id: "us.amazon.nova-lite-v2:0"
      custom_user_prompt: "coactive_user.txt"  # Required for 100-tag system
```

---

### Evaluation (Summary Evaluation Task)

LLM-as-a-judge configuration.

```yaml
evaluation:
  generated_summaries_file: "data/outputs/summaries_gpt-5_1.jsonl"
  ground_truth_file: "data/inputs/summarization_ground_truth.jsonl"

  criteria:
    - name: "factual_accuracy"
      weight: 1.0
      description: "Accuracy of facts and details"

  aggregation_method: "sum"
```

- `criteria`: List of evaluation criteria (each scored 1-5)
- `aggregation_method`: `sum` (e.g., 4-20 for 4 criteria) or `average` (1-5 scale)

---

## Pipeline Components

Components are task-specific. All model-specific parameters are detailed in the [Model Reference Guide](MODEL_REFERENCE.md).

### Preprocessor

Converts videos to frames. **Required** for frame-based models; **omit** for native video API models.

```yaml
pipeline:
  preprocessor:
    type: "frame_sampling"
    config:
      x_frames: 32
      output_format: "base64"  # or "bytes" for NVIDIA
      run_id_for_cache: null   # Optional: reuse from previous run
      storage:
        type: "s3"             # or "gcs", "local"
        config: {}             # Storage-specific params
```

**Key Parameters**:
- `x_frames`: Number of frames to sample
- `output_format`: `base64` (GPT, Qwen) or `bytes` (NVIDIA)
- `run_id_for_cache`: Reuse cached frames from previous run
- `storage`: S3/GCS/Local backend for caching

---

### Tagger (Tagging Tasks)

```yaml
pipeline:
  tagger:
    type: <registry_key>  # See Model Reference
    config:
      model_id: <model_id_or_arn>  # Model ID or inference profile ARN (Bedrock only)
      model_name: <model_name>
      inference_params:  # Optional: model-specific inference parameters
        # Parameters vary by model - see Model Reference Guide
```

**Registry Keys**: `anthropic_vision`, `bedrock_vision`, `openai_vision`, `vertex_vision`, `qwen_vision`

**Inference Parameters**: All models except Bedrock support an optional `inference_params` dict for model-specific parameters (e.g., `max_tokens`, `temperature`, `reasoning` for OpenAI). Parameters are passed directly to the model API. Bedrock support will come in a future update. See [Model Reference Guide](MODEL_REFERENCE.md) for model-specific options.

**Note**: For Bedrock models, `model_id` can be either a model ID (e.g., `"us.amazon.nova-lite-v2:0"`) or an inference profile ARN for cost tracking purposes. See [Cost Calculation Guide](COST_CALCULATION_GUIDE.md) for details on creating tagged inference profiles.

**All parameters are model-specific**. See [Model Reference Guide](MODEL_REFERENCE.md) for complete config examples.

---

### Summarizer (Summarization Tasks)

```yaml
pipeline:
  summarizer:
    type: <registry_key>  # See Model Reference
    config:
      model_id: <model_id>
      model_name: <model_name>
      inference_params:  # Optional: model-specific inference parameters
        # Parameters vary by model - see Model Reference Guide
```

**Registry Keys**: `anthropic_summarizer`, `bedrock_summarizer`, `openai_summarizer`, `vertex_summarizer`, `qwen_summarizer`

**Inference Parameters**: All models except Bedrock support an optional `inference_params` dict for model-specific parameters. Bedrock support will come in a future update. See [Model Reference Guide](MODEL_REFERENCE.md) for model-specific options.

**All parameters are model-specific**. See [Model Reference Guide](MODEL_REFERENCE.md) for complete config examples.

---

### Judge (Summary Evaluation Task)

```yaml
pipeline:
  judge:
    type: "openai_judge"
    config:
      model_id: <model_id>
      model_name: <model_name>
      inference_params:  # Optional: model-specific inference parameters
        # Parameters vary by model - see Model Reference Guide
```

**Currently Supported**: `openai_judge` only

**Inference Parameters**: Supports an optional `inference_params` dict for model-specific parameters (e.g., `reasoning` for OpenAI). See [Model Reference Guide](MODEL_REFERENCE.md) for options.

---

### Result Writer

```yaml
pipeline:
  result_writer:
    type: "jsonl"
    config:
      output_path: "data/outputs/predictions.jsonl"
```

**Type**: `jsonl` (only supported type)

---

### Metrics (Standard Tagging Task)

```yaml
pipeline:
  metrics:
    type: "multilabel_classification"
    config:
      pred_key: "video_level_tags"
      gt_key: "tags"
      exclude_tags: []
      exclude_failures: true
```

- `pred_key`: Key for predictions in output
- `gt_key`: Key for ground truth labels
- `exclude_tags`: Tags to ignore in metrics
- `exclude_failures`: Skip failed videos

---

## Complete Config Examples

For complete working examples, see the test configs in each task directory:

- **Standard Tagging (GPT)**: `configs/standard_tagging/testing/config_gpt_test.yaml`
- **Standard Tagging (Anthropic)**: `configs/standard_tagging/testing/config_anthropic_test.yaml`
- **Workload Benchmark (Pegasus)**: `configs/workload/testing/config_workload_bedrock_pegasus_1_2_test.yaml`
- **Workload Benchmark (Anthropic)**: `configs/workload/testing/config_workload_anthropic_test.yaml`
- **Summarization (NVIDIA)**: `configs/summarization/testing/config_bedrock_nvidia_test.yaml`
- **Summarization (Anthropic)**: `configs/summarization/testing/config_anthropic_test.yaml`
- **Summary Evaluation (Gemini)**: `configs/summary_evaluation/testing/config_openai_judge_gemini_2_5_pro_test.yaml`

These test configs demonstrate all configuration sections with working values for quick testing on small video sets.

---

## Common Config Patterns

### Switching Models

Change `type` (registry key) and update `config` with model-specific parameters:

```yaml
tagger:
  type: "bedrock_vision"  # or "anthropic_vision", "openai_vision", "vertex_vision", "qwen_vision"
  config: {}              # See Model Reference Guide
```

**Important**: Add `preprocessor` section if switching to frame-based model (GPT, NVIDIA, Qwen).

---

### Frame Caching

Reuse preprocessed frames across runs:

```yaml
# First run - frames are cached automatically
preprocessor:
  config:
    x_frames: 32
    storage: {}  # Configure S3/GCS/Local
# Output includes: "run_id": "gpt-5_1_251215_011357"

# Subsequent runs - reuse cached frames
preprocessor:
  config:
    x_frames: 32
    run_id_for_cache: "gpt-5_1_251215_011357"
    storage: {}  # Must match first run
```

**Important**: `x_frames` and `storage` config must match the original run.

---

### Testing Subset

For quick testing, use `fixed` mode with small set:

```yaml
video_selection:
  mode: "fixed"
  fixed_videos:
    - 'vid_test1.mp4'
    - 'vid_test2.mp4'
    - 'vid_test3.mp4'
```

---

## For More Information

For additional details on models, cost tracking, and getting started, please refer to:

- **[Model Reference Guide](MODEL_REFERENCE.md)** - Model-specific requirements, limitations, and best practices
- **[Cost Calculation Guide](COST_CALCULATION_GUIDE.md)** - Tracking and calculating costs for AWS, GCP, and model inference
- **[README](../README.md)** - Getting started guide and quick start examples
