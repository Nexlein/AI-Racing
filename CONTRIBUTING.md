# Contributing to AI Racing

We welcome contributions! To ensure high-quality code and maintain reproducibility, please follow these guidelines.

## 1. Setup Your Environment

Clone the repository and install the dependencies exactly as specified:

```bash
uv sync                  # Python 3.12, dependencies and CPU-only torch
uvx pre-commit install   # Ruff, secret scan and commit message checks
```

## 2. Formatting & Linting

We enforce strict formatting rules using `ruff`. If you submit code that fails these checks, the CI pipeline will automatically reject your Pull Request.

Before pushing your code, run:

```bash
# Auto-fix linting errors (Imports, simplify rules, etc.)
uv run ruff check --fix .

# Auto-format code
uv run ruff format .
```

## 3. Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Ensure your code passes all `ruff` checks.
3. Open a Pull Request and fill out the provided template.
