"""Experiment-independent checks for flat dictionaries of dense tensors
"""
import math

import torch


class ValidationError(AssertionError):
    pass


def assert_finite(results: dict[str, torch.Tensor]) -> None:
    if not isinstance(results, dict) or not results:
        raise ValidationError("results must be a nonempty dictionary")
    for name, tensor in results.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValidationError("results must map string names to tensors")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValidationError(f"{name}: only dense, non-quantized tensors are supported")
        if not bool(torch.isfinite(tensor).all()):
            raise ValidationError(f"{name}: contains NaN or infinity")


def _check_pair(expected, actual):
    assert_finite(expected)
    assert_finite(actual)
    if expected.keys() != actual.keys():
        raise ValidationError(
            f"result keys differ: missing={sorted(expected.keys() - actual.keys())}, "
            f"extra={sorted(actual.keys() - expected.keys())}"
        )
    for name, left in expected.items():
        right = actual[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValidationError(
                f"{name}: metadata differs: {tuple(left.shape)}/{left.dtype} "
                f"vs {tuple(right.shape)}/{right.dtype}"
            )


def assert_exact(expected: dict, actual: dict) -> None:
    """Compare logical tensor bytes, including the sign of zero.

    Nonfinite values are rejected even when their bytes are identical.
    """
    _check_pair(expected, actual)
    for name, left in expected.items():
        right = actual[name]
        left = left.detach().cpu().resolve_conj().resolve_neg().contiguous()
        right = right.detach().cpu().resolve_conj().resolve_neg().contiguous()
        a = left.reshape(-1).view(torch.uint8)
        b = right.reshape(-1).view(torch.uint8)
        if not torch.equal(a, b):
            byte = int(torch.nonzero(a != b, as_tuple=False)[0, 0])
            element = byte // left.element_size()
            raise ValidationError(
                f"{name}: bytes differ at flat element {element}: "
                f"{left.reshape(-1)[element].item()!r} vs "
                f"{right.reshape(-1)[element].item()!r}"
            )


def assert_close(expected: dict, actual: dict, *, rtol: float, atol: float) -> None:
    """Require |actual - expected| <= atol + rtol * |expected|.

    Floating/complex tensors use tolerances; integer/bool tensors stay exact.
    """
    if not all(math.isfinite(t) and t >= 0 for t in (rtol, atol)):
        raise ValueError("tolerances must be finite and nonnegative")
    _check_pair(expected, actual)
    for name, left in expected.items():
        right = actual[name]
        left, right = left.detach().cpu(), right.detach().cpu()
        try:
            if left.is_floating_point() or left.is_complex():
                torch.testing.assert_close(right, left, rtol=rtol, atol=atol)
            else:
                if not torch.equal(left, right):
                    raise AssertionError("integer or boolean values differ")
        except AssertionError as exc:
            raise ValidationError(f"{name}: {exc}") from exc
