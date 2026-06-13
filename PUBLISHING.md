# Publishing LoopGym to PyPI

## One-time setup

1. Create a project at [pypi.org](https://pypi.org/) named **`loopgym`** (or adjust `pyproject.toml` if taken).
2. Add **`PYPI_API_TOKEN`** to this repo: Settings → Secrets → Actions.
3. Optional: configure [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) with GitHub Actions.

## Publish

Create a GitHub Release (tag `v0.1.0` → publishes `0.1.0`):

```bash
git tag v0.1.0
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes "Initial public release"
```

Or run **Actions → Publish to PyPI → Run workflow** manually.

## Install

```bash
pip install loopgym
```

## Verify locally before release

```bash
pip install build
python -m build
pip install dist/loopgym-*.whl
loopgym --help
pytest tests/ -q
```
