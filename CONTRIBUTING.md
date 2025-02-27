# Contributing to SafeContext

Welcome! 🎉 Thank you for considering contributing to SafeContext. This project aims to **sanitize and filter directives in user-uploaded documents before they reach an LLM**, ensuring a safer and more robust AI pipeline. We appreciate contributions of all kinds—whether it’s fixing bugs, improving the classifier, adding new directive patterns, or enhancing documentation.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Guidelines](#code-guidelines)
4. [Logging & Commenting](#logging--commenting)
5. [Testing](#testing)
6. [Submitting a Pull Request](#submitting-a-pull-request)
7. [Getting Help](#getting-help)

---

## Getting Started

### 1️⃣ Fork the Repository
Click the **Fork** button on the top-right of this repo to create your own copy.

### 2️⃣ Clone Your Fork
```bash
git clone https://github.com/YOUR-USERNAME/SafeContext.git
cd SafeContext
```

### 3️⃣ Set Up a Virtual Environment (Recommended)
To prevent dependency conflicts:

```bash

python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Development Setup
Before adding features or fixes, ensure you have the latest changes:

```bash

git checkout main
git pull origin main
git checkout -b your-feature-branch
```

### Installing Dependencies
Make sure you have all required packages installed:

```bash

pip install -r requirements.txt
```

## Code Guidelines
✅ **General Rules**
- **Follow PEP 8** for Python code styling (`flake8` can help).
- **Keep functions modular**—avoid long, complex methods.
- **Use type hints** to improve clarity:

```python

def classify_text(input_text: str) -> bool:
```

- Use descriptive variable names—avoid `x`, `y`, `temp`.

## Logging & Commenting
✅ **Logging Best Practices**
Extensive logging ensures SafeContext can be debugged easily. Use the centralized logger:

```python

from utils.logger import SafeContextLogger
logger = SafeContextLogger(__name__)

logger.info("Processing started")
logger.warning("Potential directive detected")
logger.error("Classification failed", exc_info=True)
```

**INFO**: General updates.
**WARNING**: Suspicious activity.
**ERROR**: Something failed.

✅ **Commenting Expectations**
- **Every function should have a docstring explaining its purpose**:

```python

def sanitize_text(input_text: str) -> str:
    """
    Removes directive phrases while keeping factual content intact.
    Args:
        input_text (str): The raw text input.
    Returns:
        str: Sanitized text without directives.
    """
```

- **Complex logic should have inline comments**.

## Testing
✅ **Run Tests Before Submitting**
SafeContext includes unit tests to ensure functionality remains intact.
Run all tests before submitting a PR:

```bash

pytest tests/
```

To test a single module:

```bash

pytest tests/test_pipeline.py
```

✅ **Write New Tests**
If adding a feature or fixing a bug, include a test case in `tests/`.
We use **pytest** and **unittest.mock** for testing.

## Submitting a Pull Request
✅ **Before Submitting**
1. Ensure your branch is up to date:

```bash

git fetch origin
git rebase origin/main
```

2. Run tests and ensure everything passes.
3. Check for linting errors with:

```bash

flake8 safecontext/
```

✅ **Open a Pull Request (PR)**
1. Push your branch:

```bash

git push origin your-feature-branch
```

2. Open a PR on GitHub from `your-feature-branch` → `main`.
3. **Include a clear description**:
  - What problem does this solve?
  - What tests were added or updated?
  - Any potential edge cases?
✅ **Code Review Process**
- One or more maintainers will review your PR.
- If changes are requested, update your branch:

```bash

git add .
git commit --amend --no-edit
git push origin your-feature-branch --force
```
- Once approved, it will be merged! 🎉

## Getting Help
If you’re unsure about anything:

- Check the Issues tab.
- Open a discussion.
- Tag `@maintainers` in a PR for feedback.

Thank you for helping make SafeContext better! 🚀