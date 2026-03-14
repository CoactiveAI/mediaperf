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

# Bedrock inference parameters
BEDROCK_MAX_TOKENS = 1000
BEDROCK_TEMPERATURE = 0.1
BEDROCK_TOP_P = 0.9
BEDROCK_TOP_K = 20

# Vertex AI inference parameters
VERTEX_MAX_OUTPUT_TOKENS = 2048
VERTEX_TEMPERATURE = 0.3
VERTEX_TOP_P = 0.1
