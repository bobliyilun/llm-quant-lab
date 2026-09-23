# llm-quant-lab

Experimental, work-in-progress implementations of LLM weight quantization primitives. The current baseline is a dependency-free symmetric per-tensor quantizer for signed integer formats, with deterministic quality metrics and tests.

## Current scope

- INT4 and INT8 symmetric quantization
- per-channel symmetric quantization for 2D weight matrices
- deterministic synthetic weight generation
- reconstruction MSE and maximum-error reporting
- zero-vector and input-validation coverage

This is a learning and benchmarking lab, not a production inference library.

## Run

```bash
python3 quantize.py --bits 4 --size 4096 --seed 7
python3 quantize.py --bits 4 --size 4096 --seed 7 --clip-percentile 99.5
python3 quantize.py --bits 4 --size 4096 --seed 7 --group-size 128
python3 quantize.py --bits 4 --size 4096 --seed 7 --benchmark-clipping
python3 quantize.py --bits 4 --seed 7 --benchmark-sizes
python3 quantize.py --size 4096 --seed 7 --calibration-stats
python3 quantize.py --bits 4 --rows 8 --size 512 --seed 7
python3 -m unittest -v
```

Passing `--rows` greater than one creates a seeded matrix with row-dependent
scales and reports reconstruction error for per-tensor and per-channel modes.
`--clip-percentile` clips absolute values with a deterministic nearest-rank
threshold before symmetric quantization; the JSON report includes that threshold.
`--benchmark-clipping` compares 90th, 95th, 99th, and 100th percentile clipping
across seeded Gaussian and Laplace weight samples.
`--benchmark-sizes` compares the same seeded Gaussian distribution at 64, 256,
1,024, and 4,096 elements so reconstruction error can be inspected by tensor size.
`--group-size` gives each contiguous group an independent symmetric scale and
reports the group count and scales used for reconstruction.
Compression reports include quantized payload plus 32-bit floating-point scale
metadata, so smaller groups and per-channel quantization show their real storage
tradeoff.
Packed INT4 values use two's-complement nibbles, with the first value in each
byte's low nibble; `pack_int4` and `unpack_int4` provide a lossless reference
encoding for quantized values.
Packed INT2 values use two-bit two's-complement pairs, with the first value in
each byte's low pair; `pack_int2` and `unpack_int2` provide the corresponding
lossless reference encoding.

`quantized_dot_product` is a reference kernel that accumulates integer products
and applies the two symmetric scales once to produce a floating-point result.
`dequantize_first_dot_product` provides the equivalent reference calculation
after separately dequantizing both inputs, for fused-kernel correctness checks.

`--calibration-stats` exports deterministic min, max, mean, population standard
deviation, and maximum magnitude as JSON for the seeded tensor.

Deterministic clipping benchmark snapshots live in `benchmarks/`; the regression
suite compares the implementation to the recorded seeded result. Each snapshot
records the command and environment that generated it.

## Numerical limitations and failure modes

This repository is a reference implementation for small, finite Python inputs;
it is not a calibrated production quantization pipeline. Treat the reported MSE
and maximum absolute error as reconstruction diagnostics, not as model-quality
or task-accuracy guarantees.

- Symmetric quantization uses one signed scale per tensor, row, or contiguous
  group and reserves the most-negative signed integer code. Distributions with
  a non-zero mean can therefore use the available code range less efficiently
  than affine quantization. The affine helper is provided for comparison, but
  the CLI's per-channel and groupwise paths are symmetric only.
- A single large magnitude determines an unclipped symmetric scale. That can
  make small weights round to zero; conversely, percentile clipping improves
  precision for the retained body while introducing bounded saturation error
  for the clipped tail. Do not select a percentile from MSE alone: inspect both
  MSE and `max_abs_error` on representative weights.
- Per-channel and smaller group sizes often reduce reconstruction error, but
  require one FP32 scale per row or group. Their compression estimates include
  scale metadata, yet do not include real tensor headers, alignment, zero-point
  metadata, kernel packing requirements, or activation storage.
- The quantized dot-product function is a correctness reference, not a hardware
  kernel. Python integers do not model accumulator width, overflow, SIMD
  instructions, cache behavior, mixed-precision accumulation, or backend
  rounding. Validate an intended inference backend independently.
- Inputs must be finite. The code rejects NaN and infinities rather than
  defining a serialization or recovery policy for them. It also does not cover
  activation quantization, per-layer calibration, bias handling, outlier
  routing, non-linear operations, or end-to-end model evaluation.

Reproduce the seeded clipping trade-off used by the regression suite with:

```bash
python3 quantize.py --bits 4 --size 256 --seed 7 --benchmark-clipping
```

For the bundled Laplace sample, 99th-percentile clipping lowers MSE versus no
clipping while raising maximum absolute error because tail values saturate. The
checked-in snapshot records the exact values and its generation environment;
rerun the command and compare its JSON output before generalizing the result.

## Final seeded comparison

The table below is a reproducible, reconstruction-only comparison. The first
four rows use the same 4,096-element Gaussian sample (`seed=7`); the final row
uses an 8-by-512 matrix whose rows intentionally have different scales, so it
compares per-tensor and per-channel quantization on the same matrix. Storage
ratios include FP32 scale metadata, but not tensor headers or alignment.

| Configuration | MSE | Max absolute error | Compression ratio | Reproduce |
| --- | ---: | ---: | ---: | --- |
| INT4, per-tensor | 0.00503008 | 0.12397314 | 7.984x | `python3 quantize.py --bits 4 --size 4096 --seed 7` |
| INT8, per-tensor | 0.00001528 | 0.00683244 | 3.996x | `python3 quantize.py --bits 8 --size 4096 --seed 7` |
| INT4, 99th-percentile clipping | 0.00298404 | 0.45344334 | 7.984x | `python3 quantize.py --bits 4 --size 4096 --seed 7 --clip-percentile 99` |
| INT4, groups of 128 | 0.00323526 | 0.12371798 | 7.529x | `python3 quantize.py --bits 4 --size 4096 --seed 7 --group-size 128` |
| INT4 matrix: per-tensor / per-channel | 0.23870605 / 0.10273656 | 0.85813270 / 0.85684416 | 7.984x / 7.877x | `python3 quantize.py --bits 4 --rows 8 --size 512 --seed 7` |

For this seeded sample, INT8 cuts reconstruction error at roughly half the
payload compression of INT4. INT4 clipping and groupwise scaling lower MSE,
but clipping makes the worst-case error larger and groupwise scaling adds
metadata. On the intentionally scale-skewed matrix, per-channel scaling more
than halves MSE while slightly reducing the estimated compression ratio.

The clipping sweep is reproducible independently and shows why percentile
selection must consider both metrics:

```bash
python3 quantize.py --bits 4 --size 256 --seed 7 --benchmark-clipping
```

See [ROADMAP.md](ROADMAP.md) for planned experiments and acceptance criteria.
