import json
import math
from pathlib import Path
import unittest

from quantize import (
    benchmark_clipping,
    benchmark_tensor_sizes,
    calibration_statistics,
    compare_matrix_quantization,
    compression_estimate,
    dequantize,
    dequantize_affine,
    dequantize_first_dot_product,
    dequantize_groupwise,
    dequantize_per_channel,
    error_metrics,
    pack_int2,
    pack_int4,
    quantize_affine,
    quantize_per_channel,
    quantize_groupwise,
    quantize_symmetric_clipped,
    quantize_symmetric,
    quantized_dot_product,
    unpack_int4,
    unpack_int2,
)


class QuantizationTests(unittest.TestCase):
    def test_tensor_size_benchmark_matches_deterministic_snapshot(self):
        snapshot_path = Path(__file__).with_name("benchmarks") / "int4_size_sensitivity_seed7.json"
        snapshot = json.loads(snapshot_path.read_text())
        self.assertEqual(
            benchmark_tensor_sizes(
                snapshot["result"]["bits"], snapshot["result"]["sizes"], snapshot["result"]["seed"]
            ),
            snapshot["result"]["tensor_sizes"],
        )

    def test_clipping_benchmark_matches_deterministic_snapshot(self):
        snapshot_path = Path(__file__).with_name("benchmarks") / "int4_clipping_seed7_size256.json"
        snapshot = json.loads(snapshot_path.read_text())
        self.assertEqual(
            benchmark_clipping(
                snapshot["result"]["bits"],
                snapshot["result"]["elements"],
                snapshot["result"]["seed"],
                snapshot["result"]["percentiles"],
            ),
            snapshot["result"]["distributions"],
        )

    def test_calibration_statistics_are_deterministic_and_complete(self):
        self.assertEqual(
            calibration_statistics([-2.0, 0.0, 2.0]),
            {"elements": 3, "min": -2.0, "max": 2.0, "mean": 0.0, "stddev": math.sqrt(8 / 3), "max_abs": 2.0},
        )
        with self.assertRaises(ValueError):
            calibration_statistics([])

    def test_quantized_dot_product_accumulates_integer_products(self):
        left, left_scale = quantize_symmetric([-1.0, 0.0, 1.0], 4)
        right, right_scale = quantize_symmetric([0.5, -0.5, 1.0], 4)
        self.assertEqual(
            quantized_dot_product(left, left_scale, right, right_scale), 3 / 7
        )
        with self.assertRaises(ValueError):
            quantized_dot_product(left, left_scale, right[:-1], right_scale)

    def test_fused_dot_product_matches_dequantize_first_reference(self):
        left, left_scale = quantize_symmetric([-1.0, -0.2, 0.3, 1.0], 4)
        right, right_scale = quantize_symmetric([0.5, -0.4, 0.7, -0.1], 4)
        self.assertAlmostEqual(
            quantized_dot_product(left, left_scale, right, right_scale),
            dequantize_first_dot_product(left, left_scale, right, right_scale),
        )
        with self.assertRaises(ValueError):
            dequantize_first_dot_product(left, left_scale, right[:-1], right_scale)

    def test_int4_packing_round_trip_including_odd_length(self):
        values = [-8, -7, -1, 0, 1, 7, -3]
        packed = pack_int4(values)
        self.assertEqual(packed, bytes((0x98, 0x0F, 0x71, 0x0D)))
        self.assertEqual(unpack_int4(packed, len(values)), values)
        with self.assertRaises(ValueError):
            pack_int4([8])
        with self.assertRaises(ValueError):
            unpack_int4(packed, len(values) + 2)

    def test_int2_packing_round_trip_including_partial_byte(self):
        values = [-2, -1, 0, 1, -2, 1]
        packed = pack_int2(values)
        self.assertEqual(packed, bytes((0x4E, 0x06)))
        self.assertEqual(unpack_int2(packed, len(values)), values)
        with self.assertRaises(ValueError):
            pack_int2([2])
        with self.assertRaises(ValueError):
            unpack_int2(packed, len(values) + 3)

    def test_zero_tensor_round_trip(self):
        packed, scale = quantize_symmetric([0.0, 0.0], 4)
        self.assertEqual(packed, [0, 0])
        self.assertEqual(dequantize(packed, scale), [0.0, 0.0])

    def test_int8_has_no_more_error_than_int4(self):
        values = [-1.0, -0.37, 0.12, 0.44, 1.0]
        errors = []
        for bits in (4, 8):
            packed, scale = quantize_symmetric(values, bits)
            errors.append(error_metrics(values, dequantize(packed, scale))["mse"])
        self.assertLessEqual(errors[1], errors[0])

    def test_affine_quantization_tracks_zero_point(self):
        values = [-1.0, 0.0, 1.0]
        packed, scale, zero_point = quantize_affine(values, 8)
        self.assertEqual((packed, zero_point), ([0, 128, 255], 128))
        self.assertEqual(dequantize_affine(packed, scale, zero_point)[1], 0.0)
        self.assertLess(error_metrics(values, dequantize_affine(packed, scale, zero_point))["mse"], 0.00002)

    def test_per_channel_quantization_uses_independent_scales(self):
        matrix = [[-1.0, -4 / 7, 4 / 7, 1.0], [-8.0, -32 / 7, 32 / 7, 8.0]]
        packed, scales = quantize_per_channel(matrix, 4)
        self.assertEqual(scales, [1 / 7, 8 / 7])
        self.assertEqual(packed, [[-7, -4, 4, 7], [-7, -4, 4, 7]])
        self.assertEqual(dequantize_per_channel(packed, scales), matrix)

    def test_per_channel_beats_per_tensor_on_scaled_rows(self):
        matrix = [
            [-0.1, -0.05, 0.05, 0.1],
            [-4.0, -2.0, 2.0, 4.0],
        ]
        comparison = compare_matrix_quantization(matrix, 4)
        self.assertLess(
            comparison["per_channel"]["mse"], comparison["per_tensor"]["mse"]
        )

    def test_groupwise_quantization_uses_independent_scales_and_partial_group(self):
        values = [-1.0, -4 / 7, 4 / 7, 1.0, -8.0, 8.0]
        packed, scales = quantize_groupwise(values, 4, 4)
        self.assertEqual(packed, [-7, -4, 4, 7, -7, 7])
        self.assertEqual(scales, [1 / 7, 8 / 7])
        self.assertEqual(dequantize_groupwise(packed, scales, 4), values)

    def test_compression_estimate_includes_scale_metadata(self):
        self.assertEqual(
            compression_estimate(16, 4, 2),
            {"scale_metadata_bits": 64, "storage_bits": 128, "compression_ratio": 4.0},
        )
        self.assertLess(
            compression_estimate(16, 4, 2)["compression_ratio"],
            compression_estimate(16, 4)["compression_ratio"],
        )

    def test_percentile_clipping_uses_nearest_rank_and_preserves_body_precision(self):
        values = [-1.0, -0.5, 0.0, 0.5, 1.0, 100.0]
        packed, scale, clip_value = quantize_symmetric_clipped(values, 4, 80)
        self.assertEqual(clip_value, 1.0)
        self.assertEqual(packed, [-7, -4, 0, 4, 7, 7])
        baseline, baseline_scale = quantize_symmetric(values, 4)
        self.assertLess(
            error_metrics(values[:5], dequantize(packed, scale)[:5])["mse"],
            error_metrics(values[:5], dequantize(baseline, baseline_scale)[:5])["mse"],
        )

    def test_clipping_benchmark_is_seeded_and_reports_both_distributions(self):
        result = benchmark_clipping(4, 64, 7, (95.0, 100.0))
        self.assertEqual(result, benchmark_clipping(4, 64, 7, (95.0, 100.0)))
        self.assertEqual(set(result), {"gaussian", "laplace"})
        self.assertEqual(set(result["laplace"]), {"95.0", "100.0"})
        self.assertGreaterEqual(result["gaussian"]["100.0"]["mse"], 0.0)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            quantize_symmetric([], 4)
        with self.assertRaises(ValueError):
            quantize_symmetric([1.0], 1)
        with self.assertRaises(ValueError):
            quantize_per_channel([[1.0], [1.0, 2.0]], 4)
        with self.assertRaises(ValueError):
            quantize_groupwise([1.0], 4, 0)
        with self.assertRaises(ValueError):
            dequantize_groupwise([1.0, 2.0], [1.0, 1.0], 4)
        with self.assertRaises(ValueError):
            quantize_symmetric_clipped([1.0], 4, 0)
        with self.assertRaises(ValueError):
            benchmark_clipping(4, 1, 0, ())
        with self.assertRaises(ValueError):
            benchmark_tensor_sizes(4, (), 0)
        with self.assertRaises(ValueError):
            compression_estimate(0, 4)


if __name__ == "__main__":
    unittest.main()
