# Contributing

Thanks for contributing to `codex_openai_server`.

## Before opening a PR

- keep changes focused and avoid unrelated refactors
- update [README.md](README.md) or [CHANGELOG.md](CHANGELOG.md) when behavior or publish-facing guidance changes
- do not include real secrets, local `.env` files, or Codex auth material

## Local setup

```bash
python -m pip install -e .[dev]
pre-commit run --all-files
python -m pytest
```

Python 3.14 is preferred for local development, while the package targets Python 3.10+.

## Pull requests

- describe the user-visible change clearly
- add or update tests when runtime behavior changes
- keep documentation aligned with the shipped metadata and release notes

If you believe you found a security issue, follow [SECURITY.md](SECURITY.md) instead of opening a public issue or PR first.
