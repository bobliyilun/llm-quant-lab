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


def quantize_symmetric_clipped(
    values: Sequence[float], bits: int, percentile: float
) -> Tuple[List[int], float, float]:
    """Symmetrically quantize after clipping magnitudes at a nearest-rank percentile."""
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be greater than 0 and at most 100")
    # Reuse the baseline validation and make percentile=100 exactly equivalent.
    quantize_symmetric(values, bits)
    magnitudes = sorted(abs(value) for value in values)
    rank = max(1, math.ceil(len(magnitudes) * percentile / 100))
    clip_value = magnitudes[rank - 1]
    clipped = [max(-clip_value, min(clip_value, value)) for value in values]
    quantized, scale = quantize_symmetric(clipped, bits)
    return quantized, scale, clip_value


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


def compare_matrix_quantization(matrix: Sequence[Sequence[float]], bits: int) -> dict:
    """Return reconstruction error for per-tensor and per-channel quantization."""
    if not matrix or not matrix[0]:
        raise ValueError("matrix must not be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have the same length")

    flattened = [value for row in matrix for value in row]
    per_tensor, tensor_scale = quantize_symmetric(flattened, bits)
    channel_packed, channel_scales = quantize_per_channel(matrix, bits)
    channel_restored = dequantize_per_channel(channel_packed, channel_scales)
    return {
        "per_tensor": error_metrics(flattened, dequantize(per_tensor, tensor_scale)),
        "per_channel": error_metrics(
            flattened, [value for row in channel_restored for value in row]
        ),
    }


def laplace_samples(rng: random.Random, size: int, scale: float = 0.5) -> List[float]:
    """Generate zero-centered Laplace samples without external dependencies."""
    return [
        scale * math.copysign(math.log1p(-2 * abs(rng.random() - 0.5)), rng.random() - 0.5)
        for _ in range(size)
    ]


def benchmark_clipping(
    bits: int, size: int, seed: int, percentiles: Sequence[float]
) -> dict:
    """Measure clipped reconstruction error for seeded Gaussian and Laplace weights."""
    if not percentiles:
        raise ValueError("percentiles must not be empty")
    report = {}
    for name in ("gaussian", "laplace"):
        rng = random.Random(seed)
        values = (
            [rng.gauss(0.0, 0.5) for _ in range(size)]
            if name == "gaussian"
            else laplace_samples(rng, size)
        )
        report[name] = {
            str(percentile): {
                "clip_value": clip_value,
                **error_metrics(values, dequantize(packed, scale)),
            }
            for percentile in percentiles
            for packed, scale, clip_value in [
                quantize_symmetric_clipped(values, bits, percentile)
            ]
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=range(2, 17), default=4)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clip-percentile",
        type=float,
        help="clip absolute values at this nearest-rank percentile before quantizing",
    )
    parser.add_argument(
        "--benchmark-clipping",
        action="store_true",
        help="compare clipping thresholds across seeded Gaussian and Laplace weights",
    )
    args = parser.parse_args()
    if args.size <= 0 or args.rows <= 0:
        parser.error("--size and --rows must be positive")
    if args.rows > 1 and args.clip_percentile is not None:
        parser.error("--clip-percentile is currently supported only with one row")
    if args.benchmark_clipping and (args.rows > 1 or args.clip_percentile is not None):
        parser.error("--benchmark-clipping cannot be combined with --rows or --clip-percentile")

    if args.benchmark_clipping:
        percentiles = (90.0, 95.0, 99.0, 100.0)
        print(json.dumps({
            "bits": args.bits,
            "elements": args.size,
            "percentiles": percentiles,
            "seed": args.seed,
            "distributions": benchmark_clipping(
                args.bits, args.size, args.seed, percentiles
            ),
        }, indent=2, sort_keys=True))
        return

    rng = random.Random(args.seed)
    weights = [
        rng.gauss(0.0, 0.5 * (row + 1))
        for row in range(args.rows)
        for _ in range(args.size)
    ]
    if args.rows > 1:
        matrix = [
            weights[offset : offset + args.size]
            for offset in range(0, len(weights), args.size)
        ]
        report = {
            "bits": args.bits,
            "columns": args.size,
            "rows": args.rows,
            "seed": args.seed,
            **compare_matrix_quantization(matrix, args.bits),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.clip_percentile is None:
        packed, scale = quantize_symmetric(weights, args.bits)
        clip_value = None
    else:
        packed, scale, clip_value = quantize_symmetric_clipped(
            weights, args.bits, args.clip_percentile
        )
    restored = dequantize(packed, scale)
    report = {
        "bits": args.bits,
        "elements": args.size,
        "seed": args.seed,
        "scale": scale,
        "theoretical_compression_ratio": 32 / args.bits,
        **error_metrics(weights, restored),
    }
    if clip_value is not None:
        report.update(
            clip_percentile=args.clip_percentile,
            clip_value=clip_value,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
