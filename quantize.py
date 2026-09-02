"""Small, dependency-free quantization reference implementation."""

import argparse
import json
import math
import random
from typing import List, Sequence, Tuple


def quantize_symmetric(values: Sequence[float], bits: int) -> Tuple[List[int], float]:
    if not values:
        raise ValueError("values must not be empty")
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("values must be finite")

    qmax = (1 << (bits - 1)) - 1
    max_abs = max(abs(value) for value in values)
    scale = max_abs / qmax if max_abs else 1.0
    quantized = [max(-qmax, min(qmax, round(value / scale))) for value in values]
    return quantized, scale


def quantize_affine(values: Sequence[float], bits: int) -> Tuple[List[int], float, int]:
    """Quantize values into an unsigned affine range with a zero point."""
    if not values:
        raise ValueError("values must not be empty")
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("values must be finite")

    qmax = (1 << bits) - 1
    minimum, maximum = min(values), max(values)
    scale = (maximum - minimum) / qmax if maximum != minimum else 1.0
    zero_point = max(0, min(qmax, round(-minimum / scale)))
    quantized = [max(0, min(qmax, round(value / scale) + zero_point)) for value in values]
    return quantized, scale, zero_point


def quantize_per_channel(
    matrix: Sequence[Sequence[float]], bits: int
) -> Tuple[List[List[int]], List[float]]:
    """Symmetrically quantize each row of a rectangular weight matrix."""
    if not matrix or not matrix[0]:
        raise ValueError("matrix must not be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have the same length")
    packed_and_scales = [quantize_symmetric(row, bits) for row in matrix]
    return [packed for packed, _ in packed_and_scales], [scale for _, scale in packed_and_scales]


def dequantize(values: Sequence[int], scale: float) -> List[float]:
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be positive and finite")
    return [value * scale for value in values]


def dequantize_per_channel(
    matrix: Sequence[Sequence[int]], scales: Sequence[float]
) -> List[List[float]]:
    if len(matrix) != len(scales):
        raise ValueError("matrix and scales must have the same number of rows")
    return [dequantize(row, scale) for row, scale in zip(matrix, scales)]


def dequantize_affine(values: Sequence[int], scale: float, zero_point: int) -> List[float]:
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be positive and finite")
    return [(value - zero_point) * scale for value in values]


def error_metrics(reference: Sequence[float], candidate: Sequence[float]) -> dict:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("inputs must have the same non-zero length")
    errors = [left - right for left, right in zip(reference, candidate)]
    return {
        "mse": sum(error * error for error in errors) / len(errors),
        "max_abs_error": max(abs(error) for error in errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=range(2, 17), default=4)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")

    rng = random.Random(args.seed)
    weights = [rng.gauss(0.0, 0.5) for _ in range(args.size)]
    packed, scale = quantize_symmetric(weights, args.bits)
    restored = dequantize(packed, scale)
    report = {
        "bits": args.bits,
        "elements": args.size,
        "seed": args.seed,
        "scale": scale,
        "theoretical_compression_ratio": 32 / args.bits,
        **error_metrics(weights, restored),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
