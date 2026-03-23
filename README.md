<p align="center">
    <img src="docs/assets/logo.png" alt="MediaPerf Logo" width="400"/>
  </p>

A production-ready framework to evaluate the video understanding performance of multimodal foundation models, based  on the real data and tasks that technical leaders and practitioners within the media industry are building and deploying in production.

## Key Features

- **13 vision-language models**: Benchmarking across AWS Bedrock (Nova, Pegasus, NVIDIA), Google Vertex AI (Gemini), OpenAI (GPT), and self-hosted (Qwen). See [Model Reference Guide](docs/MODEL_REFERENCE.md) for complete list and details.
- **4 task types**:
  - Standard tagging
  - Tagging and refinement workload
  - Summarization
  - Summary Evaluation
- **Config Validation**: Pydantic-based validation catches errors before expensive operations
- **Multi-Cloud Support**: AWS Bedrock, OpenAI, Google Vertex AI, Self-Hosted (Qwen)
- **Config-Driven**: Zero-code model swapping and experimentation
- **Smart Caching**: Frame reuse across runs with S3/GCS/Local storage backends
- **Comprehensive Tracking**: Token usage, API costs, timing metrics
- **LLM-as-a-Judge**: Automated summary quality evaluation
- **Extensible**: Plugin architecture for adding models, tasks, and components

## Tasks

- Video-level tagging
- Video-level summarization
- Video-level tagging and refinement workload

## Measurements

- Performance
    - Video-level tagging: precision, recall, F1
    - Video-level summarization: Rubric-based score (using LLM-as-judge evaluation)
    - Video-level tagging and refinement workload: N/A
- Cost
- Latency/throughput

## Data

As our core dataset, we use the "Automatic Understanding of Image and Video Advertisements"[[1]](#references) video data and annotations. As a brief description:

- **Video data:** 2,003 ads videos ranging in length from 30s to 2m 30s for a total duration of 1,749 minutes.
- **Annotations:** 68 video-level tags focused on topics and sentiment.

We augment this dataset with additional summaries for the same video data from human annotators. Briefly:
- Video data: Same as above
- Augmented annotations:
   - Video-level summaries focused on long-form editorial descriptions, including storyline, intent, message, tone and target audience.
   - 100 video-level tags, including genre, format, subject, mood and themes (note: planned in next MediaPerf release in Q2 2026).

### Notes/caveats

- A number of tags from the original list were omitted from analysis due to limited coverage or an incosistent application (e.g. `funny`, `effective`, `exciting`).
- The dataset YouTube video IDs are available at `data/inputs/youtube_video_ids.txt`.
- Videos should be named `vid_<youtube_id>.mp4` (e.g., `vid_8iXdsvgpwc8.mp4`) when stored in S3, GCS, or locally.
- Video-level summaries are available at `data/inputs/summarization_ground_truth.jsonl`.

## Technologies Used

- Python 3.12
- UV for dependency/environment management
- AWS Bedrock (Nova Pro v1.0, Nova Lite v2.0, Pegasus 1.2, NVIDIA Nemotron Nano 12B v2 VL)
- OpenAI API (GPT 5.1)
- Google Vertex AI (Gemini 2.5 Pro, Gemini 3.0 Pro)
- Self-Hosted Qwen (Qwen3-VL-30B-A3B-Instruct-FP8)
- Pydantic v2 for config validation
- python-dotenv for environment loading
- OpenCV for video processing
- Pytest for testing

## Architecture

The project uses a **plugin architecture** with Registry + Factory + Builder patterns for zero-code extensibility. Components are config-driven and dynamically loaded at runtime.

Key design patterns:
- **Registry Pattern**: Dynamic component registration and discovery
- **Factory Pattern**: Component construction from YAML configuration
- **Strategy Pattern**: Interchangeable storage backends (S3/GCS/Local)
- **Validation Pattern**: Pydantic-based config validation at load time

All models inherit from base classes providing consistent prompt management, response parsing, and token tracking.

**Config Validation**: All configuration files are validated using Pydantic v2 schemas before execution. This catches structural errors, missing files, invalid registry keys, and environment variable issues before any expensive operations (video downloads, API calls) occur.

## Project Structure

```
benchmarks-project/
├── src/benchmarks/           # Main package
│   ├── models/              # Taggers, Summarizers, Judges
│   │   └── prompts/         # Model-specific prompts organized by task
│   ├── tasks/               # Task orchestration
│   ├── metrics/             # Evaluation logic
│   ├── preprocessing/       # Video → frames transformation
│   ├── storage/             # S3/GCS/Local backends
│   ├── schemas/             # Pydantic validation schemas
│   │   ├── api_schemas.py   # API response schemas
│   │   └── config_schemas/  # Config validation (modular)
│   │       ├── __init__.py      # Main validate_config() entry
│   │       ├── base.py          # Base configs (logging, tracking, pricing)
│   │       ├── video.py         # Video source/selection validation
│   │       ├── storage.py       # Storage backend validation
│   │       ├── preprocessor.py  # Preprocessor + cache validation
│   │       ├── components.py    # Component validation (taggers, etc.)
│   │       ├── paths.py         # Path configs (task-specific)
│   │       ├── tasks.py         # Task-specific settings
│   │       └── task_configs.py  # Full task configurations
│   ├── utils/               # AWS, GCP, timing, config validation
│   ├── datasets/            # Ground truth loaders
│   ├── writers/             # Output formatting
│   ├── registry.py          # Component registries
│   ├── builders.py          # Factory functions
│   └── enums.py             # Type-safe enums
├── configs/                 # Task configurations
│   ├── standard_tagging/
│   ├── workload/
│   ├── summarization/
│   ├── summary_evaluation/
│   └── preprocessing/       # Preprocessing & MediaConvert configs
├── tests/                   # Test suite
│   ├── test_config_validation.py  # Integration test for all configs
│   ├── test_*.py            # Unit tests
│   └── fixtures/            # Test data and helpers
├── scripts/                 # Utility scripts
├── data/                    # Datasets and results
├── docs/                    # Documentation
└── pyproject.toml           # Project configuration
```

## Setup

1. Install UV
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository
   ```bash
   git clone <repository-url>
   cd benchmarks-project
   ```

3. Install dependencies
   ```bash
   uv sync --all-groups
   ```

4. Install pre-commit hooks
   ```bash
   uv run pre-commit install
   ```

5. Configure environment variables
   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file and add your credentials:
   ```bash
   # OpenAI
   OPENAI_API_KEY=your_openai_key_here

   # AWS (or use ~/.aws/credentials)
   AWS_ACCESS_KEY_ID=your_aws_key_id
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key

   # GCP (or use service account JSON)
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   ```

## How to Run

`main.py` is the entrypoint. Specify the config you want to run the pipeline for as an argument.

```bash
uv run main.py path/to/config.yaml
```
Example:
```bash
uv run main.py configs/standard_tagging/testing/config_openai_test.yaml
```

**Config files allow to specify all parameters needed to set up and run a pipeline**: task, model, videos, preprocessor config if needed, etc.

That's it!

## Quickly testing the pipeline end-to-end

To test the pipeline, use configs from the `testing/` subdirectories. These are configured to run on a small number of videos with debug logging enabled:

```bash
# Example: Test standard tagging
uv run python main.py configs/standard_tagging/testing/config_gpt_test.yaml

# Example: Test summarization
uv run python main.py configs/summarization/testing/config_gpt_test.yaml
```

## Running Tasks

The framework supports four task types, all executed through `main.py`:

```bash
# Standard Tagging - Multi-label classification with metrics
uv run python main.py configs/standard_tagging/config_bedrock.yaml

# Workload Benchmark - Timing and cost analysis with iterative prompts
uv run python main.py configs/workload/config_workload_gpt.yaml

# Summarization - Generate video summaries
uv run python main.py configs/summarization/config_gpt.yaml

# Summary Evaluation - LLM-as-a-judge quality assessment
uv run python main.py configs/summary_evaluation/config_openai_judge_gpt.yaml
```

**Note**: All configs are validated at load time using Pydantic schemas. Validation errors are reported with clear messages before any execution begins.

## Utility Scripts

### Collect video metadata
Extract duration, resolution, codec, and file size from videos in S3 using ffprobe. Used for duration-based video selection (e.g., selecting videos to reach a target number of minutes):
```bash
uv run python scripts/collect_video_metadata.py \
  --bucket benchmarks-project-dev \
  --prefix youtube_ads_dataset_h264/ \
  --output data/inputs/youtube_ads_video_metadata.json
```

### Frame sampling
```bash
uv run python scripts/preprocess_frames_only.py \
  --config configs/preprocessing/config_preprocess_duration_based.yaml
```

### Convert videos using AWS MediaConvert
- Always converts to H.264 codec
- Optionally downscales resolution (set `max_height` in config)
- For 720p downscaling (NVIDIA models): use config_mediaconvert_720p.yaml
- For codec conversion only: use config_mediaconvert_duration_based.yaml

```bash
uv run python scripts/convert_videos_mediaconvert.py \
  --config configs/preprocessing/config_mediaconvert_720p.yaml
```

## Testing

Run tests using `pytest`:

```bash
# Run all tests
uv run pytest

# Run only fast tests (<5 seconds)
uv run pytest -m fast

# Run specific test file
uv run pytest tests/test_response_parsing.py
```

The test suite includes test cases covering:
- **Config validation**: Integration test validates all configs
- **Model API response parsing**: JSON parsing, tag filtering, confidence handling
- **Multi-label classification metrics**: Precision, recall, F1 calculations
- **LLM-as-a-judge evaluation**: Sum/average aggregation, criterion statistics
- **Cost and timing calculations**: Per-video, per-iteration aggregation
- **Label loading and validation**: JSONL/JSON formats, threshold validation

## Code Formatting

Format code using Ruff:

```bash
# Format all Python files
uv run ruff format
```

Pre-commit hooks automatically run Ruff on staged files before each commit.

## Additional Documentation

Additional information can be found in the `docs/` directory:

- **[Model Reference Guide](docs/MODEL_REFERENCE.md)** - Model-specific requirements, limitations, and best practices

- **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)** - Complete walkthrough of config file structure and options

- **[Cost Calculation Guide](docs/COST_CALCULATION_GUIDE.md)** - Tracking and calculating costs for AWS, GCP, and model inference

## Known Limitations

MediaPerf delivers a production-ready benchmark for real media tasks today. The following are areas where future iterations can extend its coverage and value further. Contributions are welcome — whether that's code to this repo, licensed data for benchmarking, or joining our working group.

- **No model size segmentation.** Results compare models of different sizes on the same leaderboard without size-class groupings. Future iterations will introduce further models and parameter-based tiers for fairer comparison.
- **Generative VLMs only.** The benchmark currently evaluates vision-language models (e.g., Gemini, GPT, Qwen) on generative tasks (tagging, summarization). Encoder-based models used for embedding and search/retrieval workflows are not yet covered but are planned for future iterations.
- **Short-form video only.** The dataset includes short-form video only. Longer-form content (episodes, films, sports, news) is on the roadmap but not yet included. Additional licensed data is needed to expand coverage.
- **Pipeline steps not decoupled.** The pipeline does not explicitly isolate certain steps in pre-processing (e.g., frame sampling, resolution scaling), making it harder to attribute performance differences. Future iterations could introduce measurement at pipeline stages to enable comparative analysis of key optimizations.
- **Limited output standardization.** The benchmark does not enforce standardized output formats (e.g., timecodes, structured metadata) required for downstream media workflows.
- **Hardware- and platform-dependent results.** Cost and latency numbers are tied to specific cloud providers and instance types (e.g., GCP vs. AWS). Current results represent our best attempt to provide practical comparative measurements despite differences across environments.
- **Pipelines reflect typical engineering effort.** Inference pipelines were built using publicly available documentation and best practices such that they are representative of what a typical engineering team could stand up in a reasonable timeframe (not provider-specific optimizations inaccessible to most teams). Future iterations may include a provider-optimized task track, contingent on involvement from model providers and platforms.

## License

The source code in this repository is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

Our human-annotated summaries and tags are licensed under the **Creative Commons Attribution 4.0 International License (CC-BY 4.0)**. See [LICENSE-DATA](LICENSE-DATA) for details.

## References

[1] Zaeem Hussain, Mingda Zhang, Xiaozhong Zhang, Keren Ye, Christopher Thomas, Zuha Agha, Nathan Ong, Adriana Kovashka. "Automatic Understanding of Image and Video Advertisements." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 1705-1715. [Link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Hussain_Automatic_Understanding_of_CVPR_2017_paper.pdf)