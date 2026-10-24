# Neural Observatory

> First-principles deep neural network training, internal activation inspection, and memory-constrained quantization.
> Python 3.11+ | NumPy Engine & Pure-Python Mode | Interactive Browser Dashboard | Zero Framework Dependencies

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-brightgreen.svg?style=flat-square)](https://www.python.org/)
[![Tests: Passing](https://img.shields.io/badge/Tests-5%20Passed-brightgreen.svg?style=flat-square)](tests/)

---

## 1. Overview

Neural Observatory is an educational and research software suite for building, observing, and compressing multilayer perceptron (MLP) neural networks from scratch without relying on modern automatic differentiation frameworks (PyTorch, TensorFlow, or JAX).

The project enables granular introspection into internal representation learning on the MNIST handwritten digit benchmark:
- First-Principles Engine: Implements analytical forward propagation, backpropagation, and stochastic gradient descent (SGD) entirely in NumPy, with a fallback pure-Python standard library implementation (`python -S`).
- Activation and Gradient Saliency: Tracks activation distributions, dead-neuron ratios (zero activation), and gradient norms across training epochs.
- Memory-Budgeted Quantization: Evaluates FP32 vs INT8 weight-only quantization under strict memory budgets (from 210 KB down to 5 KB).
- Zero-Dependency Offline Dashboard: An all-in-one interactive HTML5/Canvas visualization interface (`artifacts/Neural_Observatory.html`) that embeds weights, activations, and interactive digit-drawing canvas for real-time in-browser inference without internet or server execution.

---

## 2. Experimental Results (MNIST Benchmark)

Evaluation over 55,000 training, 5,000 validation, and 10,000 test images across 3 random seeds (42, 43, 44):

| Architecture | Precision | Test Accuracy (Mean +/- SD) | Serialized Model Size | Inference Latency (p50) |
|---|---|---:|---:|---:|
| 784-64-32-10 / ReLU | FP32 | 97.747 +/- 0.046% | 210,730 Bytes | 3.842 ms |
| 784-64-32-10 / ReLU | INT8 | 97.750 +/- 0.050% | 53,551 Bytes | 4.790 ms |
| 784-20-10 / Sigmoid | FP32 | 95.627 +/- 0.143% | 63,770 Bytes | 1.180 ms |
| 784-20-10 / Sigmoid | INT8 | 95.613 +/- 0.136% | 16,279 Bytes | 1.376 ms |
| 784-20-10 / ReLU | FP32 | 95.850 +/- 0.217% | 63,767 Bytes | 1.112 ms |
| 784-20-10 / ReLU | INT8 | 95.850 +/- 0.243% | 16,276 Bytes | 1.405 ms |
| 784-6-10 / ReLU | FP32 | 91.060 +/- 0.695% | 19,246 Bytes | 0.340 ms |
| 784-6-10 / ReLU | INT8 | 91.050 +/- 0.637% | 5,047 Bytes | 0.410 ms |

Key Takeaways:
- INT8 weight-only quantization achieves a 3.93x storage reduction with zero accuracy degradation (< 0.01% delta).
- Compact architectures (784-20-10) fit within a 16 KB micro-controller memory budget while maintaining > 95.8% test accuracy.

---

## 3. Interactive Visualization Dashboard

Open `artifacts/Neural_Observatory.html` in any modern web browser (Chrome, Edge, Firefox, Safari).
The dashboard is self-contained with embedded base64 weights and pre-computed telemetry:
1. Layer Exploration: Inspect test samples, replay training checkpoints, inspect weight matrices, toggle individual neurons on/off, and test custom hand-drawn digits via HTML5 Canvas.
2. Learning Trajectories: Real-time curves for loss, accuracy, activation sparsity, and gradient norm evolution.
3. Budget Optimization: Compare storage efficiency vs accuracy across 8 architectural variants.
4. Diagnostics: Confusion matrices, perturbation robustness analysis, and error mode breakdowns.

---

## 4. Quick Start Guide

### Prerequisites
- Python 3.11 or higher
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/nhdnhut/neural-observatory.git
cd neural-observatory

# Install requirements
pip install -r requirements.txt
```

### Running Tests
```bash
pytest tests -v
```

### Reproducing Training Pipeline
```bash
# Verify environment and run baseline tests
python main.py test

# Train architectures across 3 random seeds
python main.py train --epochs 18 --seeds 42 43 44

# Build bundled artifacts and reports
python main.py build
```

### Pure Python Mode (Zero Dependencies)
```bash
# Run algorithmic proof without NumPy
python -S scripts/train_pure.py --samples 1024 --epochs 2
python -S scripts/infer.py --index 0
```

---

## 5. Directory Structure

```
neural-observatory/
├── artifacts/                  # Generated plots, CSV benchmarks, and standalone HTML dashboard
├── dashboard/                  # Source files for the interactive visualization UI
├── data/                       # MNIST dataset cache and verification hashes
├── docs/                       # Technical documentation and experimental specifications
├── legacy/                     # Original prototype codebase for historical provenance
├── notebooks/                  # Reproducible Jupyter notebooks for inspection and gradient checks
├── observatory/                # Core Python package (layers, optimizers, quantization, metrics)
├── scripts/                    # Pure-Python standalone demonstration scripts
├── tests/                      # Automated unit test suite
├── main.py                     # Central CLI driver (test, train, evaluate, build)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## 6. Author and License

- Author: Nguyen Huynh Dang Nhut (@nhdnhut)
- Affiliation: Faculty of Computer Science, University of Information Technology (UIT, VNU-HCM)
- License: MIT License (see [LICENSE](LICENSE))

