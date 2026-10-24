# System architecture and numerical contract

| Path | Responsibility |
|---|---|
| `legacy/` | Unmodified original script, loader, README and dependencies |
| `observatory/reference.py` | Standard-library MLP and analytical backprop |
| `observatory/engine.py` | Batched NumPy equations; manual SGD momentum |
| `observatory/dataset.py` | Checksummed downloads, IDX parsing and deterministic split |
| `observatory/packing.py` | FP32/INT8 serialization and packed inference |
| `scripts/train_suite.py` | 12-run experiment, 24 exports, raw measurements |
| `scripts/train_pure.py` | Separately labeled Python-only training |
| `scripts/build_observatory.py` | Checkpoints, ablation, robustness, figures and offline lab |
| `dashboard/` | Offline inspection; no frameworks or network calls |
| `tests/` | Numerical contracts and serialization |
| `notebooks/` | Baseline, training and inspection guides |

## Equations

Weights use `[out,in]`, inherited from the original. Batches use `[batch,in]`, so `Z = A @ W.T + b`. Hidden activations are ReLU or sigmoid. Output is stable softmax. Mean cross-entropy derivative at logits: `(P - one_hot(y)) / batch_size`.

`dW = delta.T @ A_prev`, `db = sum(delta,axis=0)`. Propagate using unmodified `W`, apply the hidden derivative, then call `step` after all gradients exist. Momentum: `v = 0.9*v + g`, `theta -= lr*v`. Predeclared cosine schedule from configured LR to 10% of it over 18 epochs.

## Binary format v1

Little endian; no pickle execution.

1. 8-byte magic: `NOBS32v1` or `NOBS08v1`.
2. uint32 length of UTF-8 JSON metadata.
3. Metadata: dimensions, activation, quantization, arithmetic, bias dtype.
4. Per layer: row-major weights, optional per-output scales, biases.

FP32 weights/biases use four bytes. INT8 weights use signed bytes in [-127,127], one float32 scale per output neuron, and float32 biases. Zero rows use scale 1. Scalar inference computes `sum(q_i*a_i)*scale + bias` without a full dequantized matrix. Arithmetic remains floating point. NumPy quality evaluation temporarily dequantizes weights and is not a compressed RAM benchmark.

## UI invariants

- Epoch controls select stored checkpoints, without interpolation.
- INT8 forces the validation-selected checkpoint.
- Layer explorer is seed 42; aggregate tables use three seeds.
- Select a neuron, then explicitly clamp it to zero. Multiple clamps can be retained/reset.
- Saliency differentiates the currently predicted **logit** through the clamped network.
- Activation and weight map scales are local and disclosed.
- Demo examples intentionally include failures and quantization changes; their composition is not an error-rate estimate.

## Statistics and limits

Mean/sample SD summarize seeds 42,43,44 on one test set; not confidence intervals over all handwriting. Configurations and selection rule were fixed before test evaluation. Architectures can also differ in activation/LR; comparisons are system comparisons, not isolated causal estimates of depth.

Budget recommendations use highest mean validation accuracy among candidates that fit the file budget. A slider is not a memory benchmark. Process RAM, allocator overhead, energy and hardware acceleration are not measured.
