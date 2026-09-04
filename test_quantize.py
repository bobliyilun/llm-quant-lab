import unittest

from quantize import (
    compare_matrix_quantization,
    dequantize,
    dequantize_affine,
    dequantize_per_channel,
    error_metrics,
    quantize_affine,
    quantize_per_channel,
    quantize_symmetric_clipped,
    quantize_symmetric,
)


class QuantizationTests(unittest.TestCase):
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

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            quantize_symmetric([], 4)
        with self.assertRaises(ValueError):
            quantize_symmetric([1.0], 1)
        with self.assertRaises(ValueError):
            quantize_per_channel([[1.0], [1.0, 2.0]], 4)
        with self.assertRaises(ValueError):
            quantize_symmetric_clipped([1.0], 4, 0)


if __name__ == "__main__":
    unittest.main()
