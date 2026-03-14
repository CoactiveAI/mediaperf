import glob
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from benchmarks.schemas.config_schemas import validate_config


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock all environment variables so tests don't fail due to missing credentials."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "mock-openai-key",
            "AWS_ACCESS_KEY_ID": "mock-aws-key",
            "AWS_SECRET_ACCESS_KEY": "mock-aws-secret",
            "GOOGLE_APPLICATION_CREDENTIALS": "mock-gcp-creds",
        },
    ):
        yield


@pytest.mark.fast
def test_all_production_configs_are_valid():
    """
    Integration test: Validate all config files in configs/ directory.

    This ensures that:
    - All production configs have valid structure
    - All required fields are present
    - Cross-field validation passes
    - File paths are correctly specified
    - Registry keys are valid

    If a config is invalid, this test will fail with details about what's wrong.
    """
    config_dir = Path(__file__).parent.parent / "configs"
    config_files = sorted(glob.glob(str(config_dir / "**/*.yaml"), recursive=True))

    assert len(config_files) > 0, "No config files found in configs/ directory"

    invalid_configs = []

    for config_path in config_files:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            validate_config(config)

        except Exception as e:
            invalid_configs.append((config_path, str(e)))

    if invalid_configs:
        error_msg = "\n\nInvalid configuration files found:\n\n"
        for path, error in invalid_configs:
            rel_path = Path(path).relative_to(Path(__file__).parent.parent)
            error_msg += f"     [INVALID] {rel_path}\n"
            error_msg += f"     {error}\n\n"
        pytest.fail(error_msg)
