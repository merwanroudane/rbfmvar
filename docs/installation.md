---
layout: default
title: Installation
---

# Installation Guide

[← Back to Home](.)

---

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux

### Dependencies

The following packages are automatically installed:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20.0 | Numerical computations |
| scipy | ≥1.7.0 | Statistical functions |
| pandas | ≥1.3.0 | Data handling |
| statsmodels | ≥0.13.0 | Econometric tools |
| matplotlib | ≥3.4.0 | Plotting |
| tabulate | ≥0.8.0 | Table formatting |

---

## Installation Methods

### Method 1: PyPI (Recommended)

The easiest way to install RBFM-VAR:

```bash
pip install rbfmvar
```

To upgrade to the latest version:

```bash
pip install --upgrade rbfmvar
```

### Method 2: From Source (GitHub)

For the latest development version:

```bash
git clone https://github.com/merwanroudane/rbfmvar.git
cd rbfmvar
pip install -e .
```

### Method 3: Direct from GitHub

```bash
pip install git+https://github.com/merwanroudane/rbfmvar.git
```

---

## Verify Installation

After installation, verify it works correctly:

```python
import rbfmvar

# Check version
print(f"RBFM-VAR version: {rbfmvar.__version__}")

# Quick test
from rbfmvar import RBFMVAR, generate_dgp

y, info = generate_dgp('case_c', T=100)
model = RBFMVAR(lag_order=1)
results = model.fit(y)

print("Installation successful!")
print(results.summary())
```

Expected output:
```
RBFM-VAR version: 2.0.0
Installation successful!
================================================================================
                          RBFM-VAR Estimation Results
...
```

---

## Optional: Development Installation

For development with testing tools:

```bash
git clone https://github.com/merwanroudane/rbfmvar.git
cd rbfmvar
pip install -e ".[dev]"
```

This installs additional packages:
- pytest (testing)
- pytest-cov (coverage)
- black (formatting)
- flake8 (linting)

Run tests:

```bash
pytest tests/ -v
```

---

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'rbfmvar'`**

Solution: Ensure you've activated the correct Python environment:
```bash
# Check Python location
which python

# Reinstall
pip install rbfmvar
```

**Issue: `ImportError: numpy.core.multiarray failed to import`**

Solution: Upgrade numpy:
```bash
pip install --upgrade numpy
```

**Issue: Permission denied**

Solution: Use `--user` flag:
```bash
pip install --user rbfmvar
```

---

## Virtual Environment (Recommended)

It's best practice to use a virtual environment:

```bash
# Create virtual environment
python -m venv rbfmvar-env

# Activate (Linux/macOS)
source rbfmvar-env/bin/activate

# Activate (Windows)
rbfmvar-env\Scripts\activate

# Install package
pip install rbfmvar
```

---

## Conda Environment

For Anaconda users:

```bash
# Create conda environment
conda create -n rbfmvar python=3.10

# Activate
conda activate rbfmvar

# Install
pip install rbfmvar
```

---

## Next Steps

- [Quick Start Guide](quickstart) - Get started in 5 minutes
- [User Guide](userguide) - Complete usage documentation
- [Examples](examples) - Detailed code examples

---

[← Back to Home](.)
