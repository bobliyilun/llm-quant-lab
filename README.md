# llm-quant-lab

Experimental, work-in-progress implementations of LLM weight quantization primitives. The current baseline is a dependency-free symmetric per-tensor quantizer for signed integer formats, with deterministic quality metrics and tests.

## Current scope

- INT4 and INT8 symmetric quantization
- deterministic synthetic weight generation
- reconstruction MSE and maximum-error reporting
- zero-vector and input-validation coverage

This is a learning and benchmarking lab, not a production inference library.

## Run

```bash
python3 quantize.py --bits 4 --size 4096 --seed 7
python3 -m unittest -v
```

See [ROADMAP.md](ROADMAP.md) for planned experiments and acceptance criteria.

