import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmarks.tasks.cost_benchmark_task import run_cost_benchmark
from benchmarks.tasks.standard_tagging_task import run_standard_tagging
from benchmarks.tasks.summarization_task import run_summarization
from benchmarks.tasks.summary_evaluation_task import run_summary_evaluation
from benchmarks.utils.config_validation import validate_and_load_config


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point - routes to appropriate task based on config."""

    load_dotenv()

    # Load configuration (from command-line arg or default)
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        # If path doesn't include directory, look in configs/
        if "/" not in config_path:
            config_path = f"configs/{config_path}"
    else:
        config_path = "configs/config_bedrock.yaml"  # Default config

    config = load_config(config_path)

    validate_and_load_config(config)

    # Create timestamp once and add to config for consistent file naming
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    config["_timestamp"] = timestamp

    # Determine task type
    task_type = config.get("task_type", "standard_tagging")

    # Map task types to short names for log files
    task_name_map = {
        "standard_tagging": "tagging",
        "cost_benchmark": "workload",
        "summarization": "summarization",
        "summary_evaluation": "evaluation",
    }
    task_name = task_name_map.get(task_type, task_type)

    # Extract model name for log file naming
    # Support tagger, summarizer, and judge configs
    if "tagger" in config["pipeline"]:
        model_name = config["pipeline"]["tagger"]["config"].get(
            "model_name", config["pipeline"]["tagger"]["config"].get("model_id", "unknown")
        )
    elif "summarizer" in config["pipeline"]:
        model_name = config["pipeline"]["summarizer"]["config"].get(
            "model_name", config["pipeline"]["summarizer"]["config"].get("model_id", "unknown")
        )
    elif "judge" in config["pipeline"]:
        model_name = config["pipeline"]["judge"]["config"].get(
            "model_name", config["pipeline"]["judge"]["config"].get("model_id", "unknown")
        )
    else:
        model_name = "unknown"

    # Setup file logging with task name prefix
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task_name}_{model_name}_{timestamp}.log"

    # Get log level from config (default: INFO)
    log_level = config.get("logging", {}).get("level", "INFO").upper()

    # Remove default handler and add custom handlers with config log level
    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    logger.add(
        log_file,
        rotation="100 MB",
        retention="30 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    logger.info(f"Logging to file: {log_file} (level: {log_level})")

    # Route to appropriate task
    if task_type == "standard_tagging":
        run_standard_tagging(config)

    elif task_type == "cost_benchmark":
        run_cost_benchmark(config)

    elif task_type == "summarization":
        run_summarization(config)

    elif task_type == "summary_evaluation":
        run_summary_evaluation(config)

    else:
        logger.error(f"Unknown task type: {task_type}")
        logger.error("Valid task types: 'standard_tagging', 'cost_benchmark', 'summarization', 'summary_evaluation'")
        sys.exit(1)


if __name__ == "__main__":
    main()
