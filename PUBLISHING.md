# Publishing LoopGym to PyPI

## One-time setup

1. Create a project at [pypi.org](https://pypi.org/) named **`loopgym`** (or adjust `pyproject.toml` if taken).
2. Configure [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) for this repo:
   - **Owner:** `KanakMalpani`
   - **Repository:** `LoopGym`
   - **Workflow:** `publish.yml`
   - **Environment:** *(leave blank)*

   Or link PyPI to GitHub under your PyPI account settings if you use that flow.

3. *(Optional fallback)* Add **`PYPI_API_TOKEN`** to Settings → Secrets → Actions if you prefer token-based publish.

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
