# Model Reference Guide

Complete reference for the 16 vision-language models benchmarked in this framework, including requirements, limitations, optimal configurations, and known issues.

---

## AWS Bedrock Models

**Registry Key**: `bedrock_vision`, `bedrock_summarizer`

### Amazon Nova Lite v2.0

**Model ID**: `us.amazon.nova-2-lite-v1:0`

**Input Requirements**:
- **Input type**: S3 URI (native video API)

**API Type**: `converse`

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "us.amazon.nova-2-lite-v1:0"
    model_name: "nova-lite-v2"
    region_name: "us-east-2"
    api_type: "converse"
```

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "from_s3"  # Use S3 URIs directly
  s3:
    bucket: "your-s3-bucket"
    prefix: "videos"
    region: "us-east-2"
```

**Preprocessing**: 
None needed (omit from config)

**Cost** (per 1M tokens):
- Input: $0.33
- Output: $2.75

---

### Amazon Nova Pro v1.0

**Model ID**: `us.amazon.nova-pro-v1:0`

**Input Requirements**:
- **Video codec**: H.264 only
- **Input type**: S3 URI (native video API)

**API Type**: `converse`

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "us.amazon.nova-pro-v1:0"
    model_name: "nova-pro-v1"
    region_name: "us-east-2"
    api_type: "converse"
```

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "from_s3"  # Use S3 URIs directly
  s3:
    bucket: "your-s3-bucket"
    prefix: "videos_h264"
    region: "us-east-2"
```

**Preprocessing**: No frame sampling preprocessor needed (omit from config). MediaConvert codec conversion to H.264 needed once.

**Cost** (per 1M tokens):
- Input: $0.80
- Output: $3.20

---

### Amazon Pegasus 1.2

**Model ID**: `us.twelvelabs.pegasus-1-2-v1:0`

**Input Requirements**:
- **Input type**: S3 URI (native video API)

**API Type**: `invoke_model`. `account_id` needed in this case.

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "us.twelvelabs.pegasus-1-2-v1:0"
    model_name: "pegasus-1-2"
    region_name: "us-east-2"
    api_type: "invoke"  # Note: "invoke", not "converse"
    account_id: "your-aws-account-id"  # Required for invoke_model S3 access
```

**Video Source Setup**: 
```yaml
video_source:
  type: "s3"
  mode: "from_s3"  # Use S3 URIs directly
  s3:
    bucket: "your-s3-bucket"
    prefix: "videos"
    region: "us-east-2"
```

**Preprocessing**: No preprocessor needed (omit from config)

**Cost** (per 1M tokens):
- Input: NA (unavailable via `invoke_model` API)
- Output: $7.50

**Known Issues**:
- Requires `invoke_model()` API instead of `converse()`
- Not benchmarked for the 100-tag task: input text token limit (~2k tokens) is insufficient to pass the full 100-tag prompt

---

### NVIDIA Nemotron Nano 12B v2 VL

**Model ID**: `nvidia.nemotron-nano-12b-v2`

**Input Requirements**:
- **Input type**: Raw bytes (NOT base64)
- **Max resolution**: 720p (1280x720)
- **Max frames**: 16 frames per request

**Note:** Max resolution and frames are 'optimal' values found through experimentation. These could be improved.

**API Type**: `converse`

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "nvidia.nemotron-nano-12b-v2"
    model_name: "nvidia-nemotron-nano-12b-v2"
    region_name: "us-east-2"
    api_type: "converse"
    input_type: "frames_bytes" # Accept frames as raw bytes instead of S3 URI
```

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "local"  # Must download and preprocess
  s3:
    bucket: "your-s3-bucket"
    prefix: "videos_720p_h264"
```

**Preprocessing**:

* Transcoding to 720p (or your optimal resolution) and H.264:
  ```bash
  uv run python scripts/convert_videos_mediaconvert.py \
    --config configs/preprocessing/config_mediaconvert_720p.yaml
  ```

* Frame sampling

  ```yaml
  preprocessor:
    type: "frame_sampling"
    config:
      x_frames: 16  # Keep ≤16
      output_format: "bytes"  # CRITICAL: Must be "bytes", not "base64"
      storage:
        type: "s3"
        config:
          bucket: "your-s3-bucket"
          prefix: "frames_720p_h264"
          region_name: "us-east-2"
  ```

**Cost** (per 1M tokens):
- Input: $0.2
- Output: $0.6

**Current Performance**:
10% + failure rate across tasks with current video resolution/number of frames sampled configuration.

**Known Issues**:
- Apparent 720p limit - higher resolutions cause failures
- Payload size limit easily exceeded with too many frames or high-quality JPEGs
- Base64 encoding not supported (unique to NVIDIA)
- Higher failure rates compared to other models

**Best Practices**:
- Use 16 frames max for reliability
- Ensure videos are transcoded to 720p before processing
- Use `output_format: "bytes"` in preprocessor config
- Monitor payload sizes if using high frame counts

---

## Google Vertex AI Models

**Registry Key**: `vertex_vision`, `vertex_summarizer`

### Gemini 2.5 Pro

**Model ID**: `gemini-2.5-pro`

**Input Requirements**:
- **Input type**: GCS URI (native video API)

**Configuration Example**:
```yaml
tagger:
  type: "vertex_vision"
  config:
    model_id: "gemini-2.5-pro"
    model_name: "gemini-2_5-pro"
    project_id: null  # Set to your GCP project ID, or null to use default
    location: "us-central1"
```

**Video Source Setup**:
```yaml
video_source:
  type: "gcs"
  mode: "from_gcs"  # Use GCS URIs directly
  gcs:
    bucket: "your-gcs-bucket"
    prefix: "videos"
```

**Preprocessing**: No preprocessor needed (omit from config)

**Authentication**:
- Set `GOOGLE_APPLICATION_CREDENTIALS` to service account JSON path

**Cost** (per 1M tokens):
- Input: $1.25
- Output: $10.00

---

### Gemini 3.0 Pro

**Model ID**: `gemini-3-pro-preview`

**Input Requirements**: Same as Gemini 2.5 Pro

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "gemini-3-pro-preview"
    model_name: "gemini-3_0-pro"
    project_id: null  # Set to your GCP project ID, or null to use default
    location: "us-central1"
```

**Video Source Setup**: Same as Gemini 2.5 Pro

**Preprocessing**: No preprocessor needed (omit from config)

**Authentication**: Same as Gemini 2.5 Pro

**Cost** (per 1M tokens):
- Input: $2.00
- Output: $12.00

**Known Issues**:
- Not benchmarked for the 100-tag task: model was deprecated and the Vertex AI endpoint was no longer reachable at the time of benchmarking

---

### Gemini 3.1 Pro

**Model ID**: `gemini-3.1-pro-preview`

**Input Requirements**: Same as Gemini 2.5 Pro

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "gemini-3.1-pro-preview"
    model_name: "gemini-3_1-pro"
    project_id: null  # Set to your GCP project ID, or null to use default
    location: "us-central1"
```

**Video Source Setup**: Same as Gemini 2.5 Pro

**Preprocessing**: No preprocessor needed (omit from config)

**Authentication**: Same as Gemini 2.5 Pro

**Cost** (per 1M tokens):
- Input: $2.00
- Output: $12.00

---

### Gemini 3.1 Flash-Lite

**Model ID**: `gemini-3.1-flash-lite-preview`

**Input Requirements**: Same as Gemini 2.5 Pro

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "gemini-3.1-flash-lite-preview"
    model_name: "gemini-3_1-flash-lite"
    project_id: null  # Set to your GCP project ID, or null to use default
    location: "us-central1"
```

**Video Source Setup**: Same as Gemini 2.5 Pro

**Preprocessing**: No preprocessor needed (omit from config)

**Authentication**: Same as Gemini 2.5 Pro

**Cost** (per 1M tokens):
- Input: $0.25
- Output: $1.50

---

## OpenAI Models

**Registry Key**: `openai_vision`, `openai_summarizer`, `openai_judge`

### GPT 5.1

**Model ID**: `gpt-5.1`

**Input Requirements**:
- **Input type**: Base64-encoded JPEG frames

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "gpt-5.1"
    model_name: "gpt-5_1"
    api_key_env: "OPENAI_API_KEY"
    inference_params:
      reasoning:
        effort: "low"  # Options: minimal, low, medium, high
```

**Preprocessing**:

Frame sampling.

```yaml
preprocessor:
  type: "frame_sampling"
  config:
    x_frames: 32
    output_format: "base64"
    storage:
      type: "s3"
      config:
        bucket: "your-s3-bucket"
        prefix: "frames"
        region_name: "us-east-2"
```

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "local"  # Download and preprocess to frames
```

**Features**:
- **Structured Output**: Enforces JSON schema via `responses.parse()` API
- **Reasoning Effort**: Configurable reasoning parameter for quality/speed tradeoff

**Cost** (per 1M tokens):
- Input: $1.25
- Output: $10.00

---

### GPT 5.4

**Model ID**: `gpt-5.4-2026-03-05`

**Input Requirements**:
- **Input type**: Base64-encoded JPEG frames

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "gpt-5.4-2026-03-05"
    model_name: "gpt-5_4"
    api_key_env: "OPENAI_API_KEY"
    inference_params:
      reasoning:
        effort: "low"  # Options: minimal, low, medium, high
```

**Preprocessing**:

Same as GPT 5.1 (frame sampling with base64 encoding).

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "local"  # Download and preprocess to frames
```

**Features**:
- **Structured Output**: Enforces JSON schema via `responses.parse()` API
- **Reasoning Effort**: Configurable reasoning parameter for quality/speed tradeoff

**Cost** (per 1M tokens):
- Input: $2.50
- Output: $15.00

---

### GPT 5 Mini

**Model ID**: `gpt-5-mini-2025-08-07`

**Input Requirements**:
- **Input type**: Base64-encoded JPEG frames

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "gpt-5-mini-2025-08-07"
    model_name: "gpt-5-mini"
    api_key_env: "OPENAI_API_KEY"
    inference_params:
      reasoning:
        effort: "low"  # Options: minimal, low, medium, high
```

**Preprocessing**:

Same as GPT 5.1 (frame sampling with base64 encoding).

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "local"  # Download and preprocess to frames
```

**Features**:
- **Structured Output**: Enforces JSON schema via `responses.parse()` API
- **Reasoning Effort**: Configurable reasoning parameter for quality/speed tradeoff

**Cost** (per 1M tokens):
- Input: $0.25
- Output: $2.00

---

### GPT 5 Nano

**Model ID**: `gpt-5-nano-2025-08-07`

**Input Requirements**:
- **Input type**: Base64-encoded JPEG frames

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "gpt-5-nano-2025-08-07"
    model_name: "gpt-5-nano"
    api_key_env: "OPENAI_API_KEY"
    inference_params:
      reasoning:
        effort: "low"  # Options: minimal, low, medium, high
```

**Preprocessing**:

Same as GPT 5.1 (frame sampling with base64 encoding).

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "local"  # Download and preprocess to frames
```

**Features**:
- **Structured Output**: Enforces JSON schema via `responses.parse()` API
- **Reasoning Effort**: Configurable reasoning parameter for quality/speed tradeoff

**Cost** (per 1M tokens):
- Input: $0.05
- Output: $0.40

---

## Anthropic Claude Models

**Registry Key**: `anthropic_vision`, `anthropic_summarizer`

### Claude Opus 4.6

**Model ID**: `claude-opus-4-6`

**Input Requirements**:
- **Input type**: Base64-encoded JPEG frames

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "claude-opus-4-6"
    model_name: "claude-opus-4.7"
    api_key_env: "ANTHROPIC_API_KEY"
```

**Preprocessing**:

Frame sampling.

```yaml
preprocessor:
  type: "frame_sampling"
  config:
    x_frames: 32
    output_format: "base64"
    storage:
      type: "s3"
      config:
        bucket: "your-s3-bucket"
        prefix: "frames"
        region_name: "us-east-2"
```

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "local"  # Download and preprocess to frames
```

**Features**:
- **Structured Output**: Enforces JSON schema via `messages.parse()` API for tagging tasks
- **High Quality**: Highest tier model for complex reasoning

**Cost** (per 1M tokens):
- Input: $5.00
- Output: $25.00

---

### Claude Sonnet 4.6

**Model ID**: `claude-sonnet-4-6`

**Input Requirements**: Same as Claude Opus 4.6

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "claude-sonnet-4-6"
    model_name: "claude-sonnet-4.6"
    api_key_env: "ANTHROPIC_API_KEY"
```

**Preprocessing**: Same as Claude Opus 4.6 (frame sampling with base64 encoding)

**Video Source Setup**: Same as Claude Opus 4.6

**Features**: Same as Claude Opus 4.6

**Cost** (per 1M tokens):
- Input: $3.00
- Output: $15.00

---

### Claude Haiku 4.5

**Model ID**: `claude-haiku-4-5` (default in `DEFAULT_ANTHROPIC_MODEL`)

**Input Requirements**: Same as Claude Opus 4.6

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    model_id: "claude-haiku-4-5"
    model_name: "claude-haiku-4.5"
    api_key_env: "ANTHROPIC_API_KEY"
```

**Preprocessing**: Same as Claude Opus 4.6 (frame sampling with base64 encoding)

**Video Source Setup**: Same as Claude Opus 4.6

**Features**: Same as Claude Opus 4.6

**Cost** (per 1M tokens):
- Input: $1.00
- Output: $5.00

---

## Self-Hosted OpenAI-Compatible Models

### NVIDIA Nemotron 3 Nano Omni (vLLM)

**Registry Key**: `openai_compatible_vision`, `openai_compatible_summarizer`

**Model**: Self-hosted NVIDIA Nemotron 3 Nano Omni via OpenAI-compatible vLLM endpoint.

**Model ID**: `model` (or whatever name was passed to `--served-model-name` when launching vLLM)

**Input Requirements**:
- **Input type**: Base64-encoded video (full video, not frames)
- **Preprocessor**: `video_base64`
- **Endpoint**: Custom OpenAI-compatible vLLM server

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    base_url: "https://your-nvidia-vllm-endpoint.com/v1"
    api_key: "not-needed"
    model_id: "model"  # Must match --served-model-name on the vLLM server
    model_name: "nvidia-nemotron-3-nano-omni"
    model_prefix: "nvidia"  # Uses nvidia_system.txt and nvidia_user.txt prompts
    inference_params:
      max_tokens: 2048
      temperature: 0.2
      extra_body:
        chat_template_kwargs:
          enable_thinking: false  # Disable reasoning for video (recommended)
        mm_processor_kwargs:
          use_audio_in_video: true  # Process audio track in videos
```

**Preprocessing**: MediaConvert codec conversion to H.264 (or VP9) needed once. Then video conversion to base64.

```yaml
preprocessor:
  type: "video_base64"
  config:
    cache_dir: "data/cache/videos_h264"
```

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "local"  # Download and preprocess
  s3:
    bucket: "your-s3-bucket"
    prefix: "youtube_ads_dataset_h264"
    region: "us-east-2"
```

**Cost**: Depends on hosting setup (compute, storage, bandwidth). Set `input_per_1m_tokens: 0.0` and `output_per_1m_tokens: 0.0` in config.

**vLLM-specific parameters** (`extra_body`):
- `enable_thinking: false` — disables chain-of-thought reasoning (recommended for video tagging)
- `use_audio_in_video: true` — enables audio track processing alongside video frames

**Tested Hardware**:
- **Instance**: AWS EC2 `g7e.8xlarge`
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)

**Recommended vLLM serve flags**:
```bash
vllm serve ... \
  --media-io-kwargs '{"video":{"num_frames":256,"fps":2}}' \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --video-pruning-rate 0.5 \
  --mamba-ssm-cache-dtype=float32 \
  --kv-cache-dtype fp8 \
  --moe-backend triton  # RTX PRO 6000 Blackwell only
```

---

### Qwen3-VL-30B-A3B-Instruct-FP8

**Registry Key**: `openai_compatible_vision`, `openai_compatible_summarizer`

**Model**: Self-hosted Qwen3-VL-30B-A3B-Instruct-FP8 via OpenAI-compatible endpoint.

**Model ID**: Any. Doesn't need a specific ID since this version only needs the endpoint URL to specify the model.

**Input Requirements**:
- **Input type**: Base64-encoded frames
- **Max frames**: Depends on server configuration
- **Endpoint**: Custom OpenAI-compatible server

**Configuration Example**:
```yaml
<task>:
  type: <task type>
  config:
    base_url: "https://your-server.com/v1"
    api_key: "your-api-key"  # or set via environment variable
    model_id: "Qwen3-VL-30B-A3B-Instruct-FP8"
    model_name: "qwen3-vl-30b"
    model_prefix: "qwen"  # Uses qwen_system.txt and qwen_user.txt prompts
```

**Preprocessing:**:

Video conversion to base64.

```yaml
preprocessor:
  type: "video_base64"
```

**Video Source Setup**:
```yaml
video_source:
  type: "s3"
  mode: "local"  # Download and preprocess
  s3:
    bucket: "your-s3-bucket"
    prefix: "videos"
    region: "us-east-2"
```

**Cost**: Depends on hosting setup (compute, storage, bandwidth)

**Tested Hardware**:
- **Cluster**: Anyscale cluster with autoscaling L40S GPU instances (48GB VRAM each), up to 10 instances

---

## Model Comparison Matrix

| Model | Input Type | Codec | Preprocessing | Cost/1M (In) | Cost/1M (Out) |
|-------|-----------|-------|---------------|--------------|---------------|
| Nova Lite v2.0 | S3 URI | Any | None | $0.33 | $2.75 |
| Nova Pro v1.0 | S3 URI | H.264 | MediaConvert once | $0.80 | $3.20 |
| Pegasus 1.2 | S3 URI | Any | None | NA | $7.50 |
| NVIDIA Nemotron Nano v2 VL | Bytes (frames) | N/A | Video downsizing + frame sampling | $0.20 | $0.60 |
| Gemini 2.5 Pro | GCS URI | Any | None | $1.25 | $10.00 |
| Gemini 3.0 Pro | GCS URI | Any | None | $2.00 | $12.00 |
| Gemini 3.1 Pro | GCS URI | Any | None | $2.00 | $12.00 |
| Gemini 3.1 Flash-Lite | GCS URI | Any | None | $0.25 | $1.50 |
| GPT 5.1 | Base64 (frames) | N/A | Frame sampling | $1.25 | $10.00 |
| GPT 5.4 | Base64 (frames) | N/A | Frame sampling | $2.50 | $15.00 |
| GPT 5 Mini | Base64 (frames) | N/A | Frame sampling | $0.25 | $2.00 |
| GPT 5 Nano | Base64 (frames) | N/A | Frame sampling | $0.05 | $0.40 |
| Claude Opus 4.6 | Base64 (frames) | N/A | Frame sampling | $5.00 | $25.00 |
| Claude Sonnet 4.6 | Base64 (frames) | N/A | Frame sampling | $3.00 | $15.00 |
| Claude Haiku 4.5 | Base64 (frames) | N/A | Frame sampling | $1.00 | $5.00 |
| NVIDIA Nemotron 3 Nano Omni (vLLM) | Base64 (video) | H.264, VP9 | MediaConvert once + video_base64 | Varies | Varies |
| Qwen3-VL-30B | Base64 (video) | Any | None | Varies | Varies |

---

## For More Information

For additional details on configuration, cost tracking, and getting started, please refer to:

- **[Configuration Guide](CONFIGURATION_GUIDE.md)** - Complete walkthrough of config file structure and options
- **[Cost Calculation Guide](COST_CALCULATION_GUIDE.md)** - Tracking and calculating costs for AWS, GCP, and model inference
- **[README](../README.md)** - Getting started guide and quick start examples
