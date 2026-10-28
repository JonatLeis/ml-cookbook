"""Test validators independently, using only small hand-written tensors."""
import unittest

import torch

from validation import ValidationError, assert_close, assert_exact, assert_finite


class ValidationTests(unittest.TestCase):
    def test_finite(self):
        assert_finite({"float": torch.tensor([0.0, -1.0]), "integer": torch.tensor([3]),
                       "empty": torch.empty(0), "bool": torch.tensor(True)})

    def test_nonfinite(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValidationError, "x"):
                assert_finite({"x": torch.tensor(value)})

    def test_contract(self):
        for result in ({}, [], {"x": 1}, {1: torch.tensor(1)},
                       {"x": torch.eye(2).to_sparse()}):
            with self.subTest(result=result), self.assertRaises(ValidationError):
                assert_finite(result)

    def test_exact_scalar_empty_and_dtypes(self):
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16,
                      torch.int64, torch.bool, torch.complex64):
            for shape in ((), (0,), (2, 3)):
                a = torch.ones(shape, dtype=dtype)
                assert_exact({"x": a}, {"x": a.clone()})

    def test_noncontiguous(self):
        a = torch.arange(12).reshape(3, 4).T
        assert_exact({"x": a}, {"x": a.contiguous()})

    def test_difference_location(self):
        with self.assertRaisesRegex(ValidationError, "weights: bytes differ at flat element 1"):
            assert_exact({"weights": torch.tensor([1., 2.])},
                         {"weights": torch.tensor([1., 3.])})

    def test_signed_zero(self):
        with self.assertRaises(ValidationError):
            assert_exact({"x": torch.tensor(0.)}, {"x": torch.tensor(-0.)})

    def test_metadata(self):
        expected = {"x": torch.ones(2)}
        for actual in ({"y": torch.ones(2)}, {"x": torch.ones(1)},
                       {"x": torch.ones(2, dtype=torch.float64)}):
            with self.subTest(actual=actual), self.assertRaises(ValidationError):
                assert_exact(expected, actual)

    def test_matching_nonfinite_is_not_valid(self):
        for check in (assert_exact, lambda a, b: assert_close(a, b, rtol=1e-3, atol=1e-3)):
            with self.assertRaises(ValidationError):
                check({"x": torch.tensor(float("nan"))}, {"x": torch.tensor(float("nan"))})

    def test_tolerance(self):
        a, b = {"x": torch.tensor([1.])}, {"x": torch.tensor([1.0001])}
        assert_close(a, b, rtol=1e-3, atol=0)
        with self.assertRaises(ValidationError):
            assert_close(a, b, rtol=1e-6, atol=0)
        with self.assertRaises(ValidationError):
            assert_exact(a, b)

    def test_tolerance_is_relative_to_expected(self):
        with self.assertRaises(ValidationError):
            assert_close({"x": torch.tensor(1.)}, {"x": torch.tensor(2.)}, rtol=.75, atol=0)

    def test_integer_comparison_does_not_lose_precision(self):
        with self.assertRaises(ValidationError):
            assert_close({"x": torch.tensor(2**60)}, {"x": torch.tensor(2**60 + 1)},
                         rtol=1, atol=1)

    def test_invalid_tolerance(self):
        for tolerance in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                assert_close({"x": torch.tensor(1.)}, {"x": torch.tensor(1.)},
                             rtol=tolerance, atol=0)


if __name__ == "__main__":
    unittest.main()
