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

See [ROADMAP.md](ROADMAP.md) for planned experiments and acceptance criteria.
