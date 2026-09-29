---
title: Installation
sidebar_position: 2
---

# Installation

FloDR is on PyPI and needs Python 3.11 or newer. It trains with PyTorch and uses the GPU when one is available.

```bash
pip install flodr
```

With uv, add it to a project with `uv add flodr` or install it into the current environment with
`uv pip install flodr`.

## GPU

pip installs the PyTorch build of the package index, which may not match your CUDA driver. For a specific CUDA
version install PyTorch first, following [pytorch.org](https://pytorch.org/get-started/locally/), and FloDR
after it.

```python
import flodr

flodr.default_device()  # "cuda" when PyTorch sees a GPU, otherwise "cpu"
```

FloDR runs on the CPU as well, several times slower. See [Performance](guide/performance.md).

## From source

For development, clone the repository and install it with its locked dependencies and the test tools.

```bash
git clone https://github.com/daenuprobst/flodr
cd flodr
uv sync --extra dev
uv run pytest
```

The tests run on the CPU and take a few minutes.
