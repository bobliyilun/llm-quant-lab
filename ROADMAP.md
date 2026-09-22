# Roadmap

Each checked item must include a runnable test or reproducible measurement.

- [x] Symmetric per-tensor INT4/INT8 quantization baseline
- [x] Deterministic reconstruction-error CLI report
- [x] Unit tests for zero tensors, validation, and precision ordering
- [x] Add asymmetric affine quantization with zero points
- [x] Add per-channel quantization for 2D weight matrices
- [x] Compare per-tensor and per-channel error on seeded matrices
- [x] Add percentile clipping for outlier-heavy distributions
- [x] Benchmark clipping thresholds across Laplace and Gaussian weights
- [x] Add group-wise quantization with configurable group size
- [x] Record scale metadata overhead in compression estimates
- [x] Add packed INT4 byte encoding and round-trip tests
- [x] Add packed INT2 byte encoding and round-trip tests
- [x] Implement a quantized dot-product reference kernel
- [x] Compare dequantize-first and fused dot-product correctness
- [x] Add calibration statistics export as JSON
- [x] Add deterministic benchmark snapshots for regression checks
- [x] Measure error sensitivity across tensor sizes
- [x] Add property-style randomized invariant checks
- [x] Document numerical limitations and failure modes
- [ ] Publish a final comparison table with reproducible commands
