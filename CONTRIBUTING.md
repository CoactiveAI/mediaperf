# Contributing to MediaPerf

Thank you for your interest in contributing to MediaPerf! We welcome contributions of all kinds — new models, bug fixes, new tasks, documentation improvements, and public datasets. Every contribution, big or small, helps make this benchmark more useful for the community.

## Before You Start

Open a [GitHub issue](https://github.com/CoactiveAI/mediaperf/issues) describing what you want to do before submitting a PR. This avoids duplicated effort and ensures the contribution fits the project direction.

## Getting Started

```bash
git clone <your-fork>
cd mediaperf
uv sync --all-groups
uv run pre-commit install
uv run pytest  # confirm everything passes
```

See the [README](README.md) for full setup instructions.

## Adding a New Model

The plugin architecture makes this straightforward — no changes to `main.py` needed.

1. Create a new class inheriting from `VideoTagger`, `VideoSummarizer`, or `LLMJudge` in `src/benchmarks/models/`
2. Add model-specific prompts in `src/benchmarks/models/prompts/` (organized by task type)
3. Register the class in `src/benchmarks/registry.py`
4. Add a production config in `configs/` and a test config in `configs/<task>/testing/`
5. Add response parsing tests in `tests/`

See existing models (e.g., `openai_compatible_vision.py`) and the [Model Reference Guide](docs/MODEL_REFERENCE.md) for reference.

## Code Standards

- **Formatting**: Ruff — runs automatically via pre-commit on staged files. Run manually with `uv run ruff format`
- **Docstrings**: 1-line max if needed; omit entirely for simple functions
- **No module-level docstrings**

## Testing

Run the full test suite before submitting:

```bash
uv run pytest
```

Add tests for all new logic wherever sensible — new models, parsing, metrics, utilities. See `tests/` for examples. We are actively working to increase test coverage across the codebase.

## Submitting a PR

- Branch off `main`
- Keep PRs focused — one feature or fix per PR
- Reference the related issue in the PR description
- Ensure all tests pass and pre-commit hooks are clean

## Dataset Guidelines

Only contribute publicly available datasets. Do not include proprietary data, references to private storage buckets, or credentials of any kind.
