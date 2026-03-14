import sys
from typing import Any, Dict

from pydantic import ValidationError

from ..schemas.config_schemas import TaskConfig, validate_config


def validate_and_load_config(config_dict: Dict[str, Any]) -> TaskConfig:
    """
    Validate configuration and provide formatted error messages on failure.

    Args:
        config_dict: Raw configuration dictionary from YAML

    Returns:
        Validated task configuration

    Raises:
        SystemExit: If validation fails (exits with code 1)
    """
    try:
        return validate_config(config_dict)
    except ValidationError as e:
        print("\n" + "=" * 80)
        print("Configuration Validation Failed!")
        print("=" * 80 + "\n")

        error_count = len(e.errors())
        print(f"Found {error_count} error(s):\n")

        for i, error in enumerate(e.errors(), 1):
            loc = ".".join(str(item) for item in error["loc"])
            msg = error["msg"]
            print(f"{i}. [{loc}]")
            print(f"   {msg}\n")

        print("=" * 80)
        print("Fix these errors and try again.")
        print("=" * 80 + "\n")
        sys.exit(1)
    except ValueError as e:
        print("\n" + "=" * 80)
        print("Configuration Validation Failed!")
        print("=" * 80 + "\n")
        print(str(e))
        print("\n" + "=" * 80)
        sys.exit(1)
