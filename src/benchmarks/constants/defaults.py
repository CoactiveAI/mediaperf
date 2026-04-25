from pathlib import Path

# Base paths
BENCHMARKS_ROOT = Path(__file__).parent.parent
PROMPTS_BASE_DIR = BENCHMARKS_ROOT / "models" / "prompts"

# Prompts directories
TAGGING_PROMPTS_DIR = PROMPTS_BASE_DIR / "tagging"
SUMMARIZATION_PROMPTS_DIR = PROMPTS_BASE_DIR / "summarization"
EVALUATION_PROMPTS_DIR = PROMPTS_BASE_DIR / "evaluation"

# Model defaults
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENAI_REASONING_EFFORT = "minimal"
DEFAULT_BEDROCK_MODEL_ID = "us.amazon.nova-lite-v1:0"
DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_VERTEX_MODEL_ID = "gemini-2.5-flash"
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"

# Bedrock inference parameters
BEDROCK_MAX_TOKENS = 1000
BEDROCK_TEMPERATURE = 0.1
BEDROCK_TOP_P = 0.9
BEDROCK_TOP_K = 20

# Vertex AI inference parameters
VERTEX_MAX_OUTPUT_TOKENS = 2048
VERTEX_TEMPERATURE = 0.3
VERTEX_TOP_P = 0.1

# Default inference params per model type
DEFAULT_ANTHROPIC_INFERENCE_PARAMS = {"max_tokens": 1024}

DEFAULT_BEDROCK_INFERENCE_PARAMS = {
    "maxTokens": BEDROCK_MAX_TOKENS,
    "temperature": BEDROCK_TEMPERATURE,
    "topP": BEDROCK_TOP_P,
}

DEFAULT_OPENAI_INFERENCE_PARAMS = {}

DEFAULT_VERTEX_INFERENCE_PARAMS = {
    "max_output_tokens": VERTEX_MAX_OUTPUT_TOKENS,
    "temperature": VERTEX_TEMPERATURE,
    "top_p": VERTEX_TOP_P,
}

DEFAULT_QWEN_INFERENCE_PARAMS = {"max_tokens": 1024, "temperature": 0.3}
