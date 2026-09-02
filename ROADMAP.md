# Roadmap

Each checked item must include a runnable test or reproducible measurement.

- [x] Symmetric per-tensor INT4/INT8 quantization baseline
- [x] Deterministic reconstruction-error CLI report
- [x] Unit tests for zero tensors, validation, and precision ordering
- [x] Add asymmetric affine quantization with zero points
- [x] Add per-channel quantization for 2D weight matrices
- [ ] Compare per-tensor and per-channel error on seeded matrices
- [ ] Add percentile clipping for outlier-heavy distributions
- [ ] Benchmark clipping thresholds across Laplace and Gaussian weights
- [ ] Add group-wise quantization with configurable group size
- [ ] Record scale metadata overhead in compression estimates
- [ ] Add packed INT4 byte encoding and round-trip tests
- [ ] Add packed INT2 byte encoding and round-trip tests
- [ ] Implement a quantized dot-product reference kernel
- [ ] Compare dequantize-first and fused dot-product correctness
- [ ] Add calibration statistics export as JSON
- [ ] Add deterministic benchmark snapshots for regression checks
- [ ] Measure error sensitivity across tensor sizes
- [ ] Add property-style randomized invariant checks
- [ ] Document numerical limitations and failure modes
- [ ] Publish a final comparison table with reproducible commands
