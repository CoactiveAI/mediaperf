def load_prompt_for_iteration(prompt_path: str) -> str:
    """
    Args:
        prompt_path: Path to prompt file for this iteration

    Returns:
        Full prompt text
    """
    with open(prompt_path, "r") as f:
        return f.read().strip()
