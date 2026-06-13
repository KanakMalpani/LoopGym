# Publishing LoopGym to PyPI

## One-time setup

1. Create a project at [pypi.org](https://pypi.org/) named **`loopgym`** (register the name before first publish).
2. **Preferred:** configure [trusted publishing](https://docs.pypi.org/trusted-publishers/) on the PyPI project:
   - **PyPI project name:** `loopgym`
   - **Owner:** `KanakMalpani`
   - **Repository name:** `LoopGym`
   - **Workflow name:** `publish.yml`
   - **Environment name:** *(leave blank)*

   Linking GitHub under your PyPI account is not enough — each project needs its own trusted publisher entry.

3. **Fallback:** add **`PYPI_API_TOKEN`** (upload scope) to Settings → Secrets → Actions on this repo.

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
