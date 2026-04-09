# Cost Calculation Guide

Complete guide for tracking and calculating costs across AWS, GCP, and model inference in the benchmarks framework.

---

## Overview

The framework tracks costs across three main areas:
1. **Model inference** (API calls for tagging/summarization)
2. **Video preprocessing** (MediaConvert, frame extraction)
3. **Compute resources** (EC2 instances for preprocessing)

---

## AWS Cost Tracking

AWS cost tracking mainly relies on tagging your resources and filtering Cost Explorer by these tags, or, for preprocessing costs, running the pipeline (or preprocess-only script) inside an EC2 instance and multiplying the instance's hourly rate by the code's run time.

**DISCLAIMER:** tags specified in the code first need to be activated inside AWS. Refer to the [documentation](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/activating-tags.html) for more information.

### Inference Costs (Bedrock)

**Using Inference Profiles with Tags**

Inference profiles allow you to track costs for specific models or experiments by tagging them at creation time.

#### Step 1: Create Tagged Inference Profile

When creating an inference profile in AWS Bedrock, add tags for cost allocation. Example:

```bash
aws bedrock create-inference-profile \
  --region us-east-2 \
  --profile-name "nova-lite-v2-benchmarks" \
  --model-source "arn:aws:bedrock:us-east-2::foundation-model/us.amazon.nova-lite-v2:0" \
  --tags project=benchmarks,model=nova-lite-v2,task=tagging
```

This will generate the inference profile and provide its ARN:

`"arn:aws:bedrock:us-east-2:123456789123:application-inference-profile/abcdeg123456"`

#### Step 2: Use Profile ARN in Config

Reference the tagged inference profile you created in the previous step in your config files:

```yaml
tagger:
  type: "bedrock_vision"
  config:
    model_id: "arn:aws:bedrock:us-east-2:123456789123:application-inference-profile/abcdeg123456"
    model_name: "nova-lite-v2"
    region_name: "us-east-2"
```

The ARN format for inference profiles:
```
arn:aws:bedrock:<region>:<account-id>:application-inference-profile/<profile-id>
```

#### Step 3: Filter Costs in AWS Cost Explorer

1. Navigate to **AWS Cost Explorer**
2. Set date range for your experiment
3. Apply filters:
   - **Service**: AWS Bedrock
   - **Tag(s)**: <your tag key = your tag value>, ...

---

### Video Preprocessing Costs (MediaConvert)

**Tracking MediaConvert Costs with Tags**

MediaConvert jobs are tagged via config files to track video preprocessing costs.

#### MediaConvert Config with Tags

In your MediaConvert config file:

```yaml
mediaconvert:
  output_prefix: "youtube_ads_dataset_720p_h264/"
  role_arn: "arn:aws:iam::443102425374:role/MediaConvertRole"
  quality: "SINGLE_PASS"
  max_bitrate: 5000000
  max_height: 720  # For downsizing

  # Cost allocation tags
  tags:
    project: "benchmarks_project_mediaconvert_720p"
    task: "video_preprocessing"
    codec: "h264"
```

**Usage**:
```bash
uv run python scripts/convert_videos_mediaconvert.py \
  --config configs/preprocessing/config_mediaconvert_720p.yaml
```

#### Filter MediaConvert Costs in Cost Explorer

Filter by tag and (optionally) by service in AWS Cost Explorer, similarly to how it's done for Bedrock costs.

---

### Frame Extraction Preprocessing Costs (EC2)

**Calculating Preprocessing Costs on EC2**

When running frame extraction or other preprocessing on EC2 instances, calculate costs based on instance hourly rate × runtime.

#### Method 1: Standalone Preprocessing Script

Run the preprocessing script on an EC2 instance:

```bash
# On EC2 instance
uv run python scripts/preprocess_frames_only.py \
  --config configs/preprocessing/config_frame_sampling_duration_based.yaml
```

**Cost Calculation**:
```
Preprocessing Cost = (Instance Hourly Rate) × (Runtime in Hours)
```

**Example**:
- Instance: `t3.xlarge` ($0.1664/hour in us-east-2)
- Runtime: 2.5 hours
- Cost: $0.1664 × 2.5 = $0.416

#### Method 2: Full Pipeline with Preprocessing

Run the entire pipeline (including preprocessing) on EC2:

```bash
# On EC2 instance
uv run python main.py configs/standard_tagging/config_gpt.yaml
```

Preprocessing time can be easily calculated from the pipeline run logs.

### S3 Storage Costs

Storage associated costs are mainly due to S3-stored video ingestion (read/download) and frame saving and reading. The easiest way to track these costs is by tagging the S3 bucket being used (or buckets if running multiple experiments simultaneously, as to correctly isolate I/O associated to each run). Costs can be then filtered by this tag and your experiment run window. If calculating costs manually, below are rates for most common I/O  operations for reference:

1. **Storage Cost**: Based on GB stored per month
   - Standard: $0.023 per GB/month
   - Intelligent-Tiering: $0.023 per GB/month (frequent access)

2. **Request Cost**:
   - PUT requests: $0.005 per 1,000 requests
   - GET requests: $0.0004 per 1,000 requests

3. **Data Transfer**:
   - In: Free
   - Out to internet: $0.09 per GB (first 10 TB/month)
   - Out to EC2 in same region: Free

---

## GCP Cost Tracking

### Inference Costs (Vertex AI)

**Using Labels for Cost Tracking**

GCP uses labels (similar to AWS tags) for cost allocation tracking.  Unlike AWS, these labels do not need to be activated beforehand.

#### Step 1: Add Labels in Config

Add labels to your Vertex AI config for cost tracking. Example:

```yaml
tagger:
  type: "vertex_vision"
  config:
    model_id: "gemini-2.5-pro"
    model_name: "gemini-2_5-pro"
    project_id: null  # Set to your GCP project ID, or null to use default from credentials
    location: "us-central1"  # GCP region for Vertex AI
    labels:  # Optional: GCP labels for cost tracking/billing organization
      project: "benchmarks_gemini_2_5_pro"
      task: "tagging"
```

**Note**: Labels specified in the config are automatically applied to API calls and appear in billing data for cost filtering.

#### Step 2: Track Costs in Cloud Billing

**Option A: Cloud Console**

1. Navigate to **Cloud Billing → Reports**
2. Set date range
3. Filter by:
   - **Service** (optionally): Vertex AI API
   - **Labels**: `project:benchmarks_gemini_2_5_pro` (or your custom label)
4. Group by: **SKU** or **Label**

**Option B: BigQuery Export**

For detailed analysis, export billing data to BigQuery (details on how to set this up can be found [here](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery)):

1. Enable BigQuery export in Cloud Billing
2. Query costs with labels. Example:

```sql
SELECT
  service.description AS service,
  SUM(cost) AS total_cost,
  SUM(IFNULL((SELECT SUM(amount) FROM UNNEST(credits)), 0)) AS total_credits,
  SUM(cost) + SUM(IFNULL((SELECT SUM(amount) FROM UNNEST(credits)), 0)) AS net_cost
FROM
  `project-id.dataset_name.gcp_billing_export_resource_v1_XXXXXX_XXXXXX_XXXXXX`
WHERE
  EXISTS(SELECT 1 FROM UNNEST(labels) WHERE key = 'project' AND value = 'benchmarks_gemini_2_5_pro')
  -- filter by more than one tag if needed
  -- AND EXISTS(SELECT 1 FROM UNNEST(labels) WHERE key = 'task' AND value = 'tagging')
  -- ...
  AND and DATE(_PARTITIONTIME) BETWEEN '2026-01-01' AND '2026-01-02'
GROUP BY 1
ORDER BY net_cost DESC;
```

**Note:** Replace `project-id`, `dataset_name`, and the table name with your actual BigQuery billing export details. The query calculates net cost including credits.

---

### Video Preprocessing Costs (GCP)

**Cloud Storage Transfer and Processing**

For video preprocessing on GCP: Add tags/labels to your GCS buckets, analogous to what is done in AWS. Below are rates for common operations for reference:

1. **GCS Storage Costs**:
   - Storage: $0.020 per GB/month (Standard)
   - Operations: Class A (writes) $0.05 per 10,000 ops, Class B (reads) $0.004 per 10,000 ops

2. **Compute Engine (Frame Extraction)**:
   - Similar to EC2, calculate: Instance hourly rate × runtime
   - Example: `n1-standard-4` ($0.15/hour in us-central1)

3. **Video Transcoding** (if not using MediaConvert): Use Transcoder API with labels for cost tracking (not currently supported by this codebase).

---

## Model Inference Cost Calculation

AWS Cost Explorer/GCP Cloud Billing (or Billing Export) are the most reliable sources to get inference, ingestion and preprocessing-related costs. If cloud resources are not fully set up, aren't an option, etc., the closest proxy for these costs would be input and output token cost calculation. This section explains how this is calculated.

**Note:** Token costs are reported as inference costs for models outside of Bedrock/Vertex (e.g. GPT, Claude).

**Shared logic across all models**

All models use the same token-based cost calculation formula.

### Token Cost Configuration

Specify costs in config files using `pricing` section:

```yaml
pricing:
  input_per_1m_tokens: 0.80   # Cost per 1M input tokens (USD)
  output_per_1m_tokens: 3.20  # Cost per 1M output tokens (USD)
```

### Cost Calculation Formula

```python
Input Cost = (Input Tokens / 1,000,000) × Input Price per 1M
Output Cost = (Output Tokens / 1,000,000) × Output Price per 1M
Total Cost = Input Cost + Output Cost
```

### Cost Output

When `return_usage: true`, in the config file, results include cost data:

```json
{
  "video_id": "vid_ABC123",
  "model": "nova-pro-v1",
  "tags": ["chocolate", "cheerful"],
  "usage": {
    "input_tokens": 15234,
    "output_tokens": 423,
    "total_tokens": 15657,
    "api_call_time_seconds": 2.34,
    "input_cost_usd": 0.0122,
    "output_cost_usd": 0.0014,
    "total_cost_usd": 0.0136
  }
}
```

### Aggregate Cost Calculation

For workload benchmarking across multiple videos:

```json
{
  "total_videos": 100,
  "successful_videos": 98,
  "failed_videos": 2,
  "total_input_tokens": 1523456,
  "total_output_tokens": 42389,
  "total_cost_usd": 1.3547,
  "avg_cost_per_video_usd": 0.0138,
  "avg_input_tokens_per_video": 15545,
  "avg_output_tokens_per_video": 433
}
```

---

## For More Information

For additional details on models, configuration, and getting started, please refer to:

- **[Model Reference Guide](MODEL_REFERENCE.md)** - Model-specific requirements, limitations, and best practices
- **[Configuration Guide](CONFIGURATION_GUIDE.md)** - Complete walkthrough of config file structure and options
- **[README](../README.md)** - Getting started guide and quick start examples
