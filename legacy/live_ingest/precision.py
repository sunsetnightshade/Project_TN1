from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def to_fixed_price(value: float | int | str, *, scale: int) -> int:
    quantized = (Decimal(str(value)) * Decimal(scale)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return int(quantized)


def from_fixed_price(value: int, *, scale: int) -> float:
    return float(Decimal(value) / Decimal(scale))
