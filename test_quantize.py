import unittest

from quantize import (
    dequantize,
    dequantize_affine,
    error_metrics,
    quantize_affine,
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

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            quantize_symmetric([], 4)
        with self.assertRaises(ValueError):
            quantize_symmetric([1.0], 1)


if __name__ == "__main__":
    unittest.main()
