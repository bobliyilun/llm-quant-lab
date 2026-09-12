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
python3 quantize.py --bits 4 --rows 8 --size 512 --seed 7
python3 -m unittest -v
```

Passing `--rows` greater than one creates a seeded matrix with row-dependent
scales and reports reconstruction error for per-tensor and per-channel modes.
`--clip-percentile` clips absolute values with a deterministic nearest-rank
threshold before symmetric quantization; the JSON report includes that threshold.
`--benchmark-clipping` compares 90th, 95th, 99th, and 100th percentile clipping
across seeded Gaussian and Laplace weight samples.
`--group-size` gives each contiguous group an independent symmetric scale and
reports the group count and scales used for reconstruction.
Compression reports include quantized payload plus 32-bit floating-point scale
metadata, so smaller groups and per-channel quantization show their real storage
tradeoff.
Packed INT4 values use two's-complement nibbles, with the first value in each
byte's low nibble; `pack_int4` and `unpack_int4` provide a lossless reference
encoding for quantized values.

See [ROADMAP.md](ROADMAP.md) for planned experiments and acceptance criteria.
