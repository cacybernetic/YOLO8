# Contributing to YOLOv8 From Scratch

Thank you for your interest in contributing! 🚀  
All contributions are welcome: bug fixes, documentation improvements, new features, code optimizations (Python or Rust), etc.

This document guides you through the contribution process.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Before You Start](#before-you-start)
- [Development Environment Setup](#development-environment-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Conventions](#code-conventions)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting a Bug](#reporting-a-bug)
- [Suggesting a Feature](#suggesting-a-feature)
- [Questions](#questions)

## Code of Conduct

This project follows a [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you agree to uphold its terms. Please read it.

## Before You Start

- Check existing [issues](https://github.com/cacybernetic/YOLO8/issues) to see if your bug or suggestion has already been reported.
- For significant changes (new feature, heavy refactoring), open an issue first to discuss with the maintainers. This avoids working on something that may not be accepted.

## Development Environment Setup

The project uses Python 3.10+ and `uv` as the package manager. An optional part is written in Rust.

1. Fork the repository and clone it locally:

```bash
git clone https://github.com/YOUR_USERNAME/YOLO8.git
cd YOLO8
```

2. Create a virtual environment and install dependencies (as described in the README):

```bash
uv venv --python 3.10
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv pip install -e .
```

3. For the Rust part (optional):

```bash
cargo build --release
```

## Contribution Workflow

1. Create a branch from `main` with an explicit name:  
   `git checkout -b feature/my-new-feature`  
   or  
   `git checkout -b fix/bug-fix-xyz`

2. Make your changes following the conventions below.

3. Test your changes:
   - Ensure training, evaluation, and inference still work on a small test dataset.
   - If you modify the Rust code, make sure it compiles and the inference examples from the README remain valid.

4. Commit your changes with a clear message (see [Commit Conventions](#commit-conventions)).

5. Push your branch and open a Pull Request targeting the `main` branch of the original repository.

6. In the PR description, explain what you did, why, and reference any related issues (e.g., `Closes #12`).

7. Maintainers will review your PR, may suggest changes, and merge it once ready.

## Code Conventions

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) as much as possible.
- Use static typing (type hints) for public function signatures.
- Document complex classes and functions with concise docstrings.
- Keep the code readable: explicit variable names, comments when the logic is non-trivial.
- The project uses `torch`, `numpy`, `opencv-python`, etc. Avoid introducing unnecessary heavy dependencies.

### Rust

- Use `rustfmt` and `clippy` before committing.
- The Rust part is an inference tool: prioritize performance and robustness (error handling, avoid unwarranted `.unwrap()`).

### Commit Conventions

Use English, structured commit messages:

```
<type>: <short description>

<optional body>
```

Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Example:

```
feat: add MixUp augmentation to dataset

Implements MixUp as described in the original YOLOv8 paper.
Configurable via train.yaml. Closes #15.
```

## Testing

The project does not yet have an automated test suite. If you add a critical function (e.g., new loss, model module), it is strongly advised to include a small test script (in a `tests/` folder or directly via a `if __name__ == "__main__":` block). Mention it in your PR.

While waiting for more formal coverage, a basic functional test consists of:
- Running one epoch of training on a few images.
- Checking that evaluation runs without crashing.
- Verifying that ONNX export produces a valid file.

## Documentation

- New features must be documented in the README if they affect command-line usage.
- Configuration parameters added to `.yaml` files must be described in the relevant README section.

## Reporting a Bug

Open an issue describing:

- Project version (or commit hash).
- Your OS and environment (Python, CUDA, Rust if relevant).
- Steps to reproduce the bug.
- Expected behavior vs. observed behavior.
- Any error messages or relevant logs.

## Suggesting a Feature

Suggestions are welcome! Open an issue with the `enhancement` tag, explaining:

- The problem your feature solves.
- A description of the proposed solution.
- Any usage example or API sketch.

## Questions

If you have a question about the code or its usage, you can contact the maintainers:

- **Author**: DOCTOR MOKIRA – dr.mokira@gmail.com
- **Maintainer**: CONSOLE ART CYBERNETIC – ca.cybernetic@gmail.com

You can also simply open an issue with the `question` tag.

Thank you for contributing to **YOLOv8 From Scratch**! ❤️
