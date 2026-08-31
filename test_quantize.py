import unittest

from quantize import dequantize, error_metrics, quantize_symmetric


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

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            quantize_symmetric([], 4)
        with self.assertRaises(ValueError):
            quantize_symmetric([1.0], 1)


if __name__ == "__main__":
    unittest.main()

