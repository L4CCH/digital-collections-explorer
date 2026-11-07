from __future__ import annotations

from dataclasses import dataclass

# 15 degrees above the horizon, where most museum photographs are taken from.
POLAR_DEGREES = 75.0

COLOR = "color"
RELIEF = "relief"
MODES = (COLOR, RELIEF)


def degrees(value: float) -> str:
    return f"{value:g}deg"


@dataclass(frozen=True)
class View:
    number: int
    theta: float
    phi: float

    @property
    def orbit(self) -> str:
        return f"{degrees(self.theta)} {degrees(self.phi)} auto"

    def file_name(self, mode: str) -> str:
        return f"{self.number:02d}-{mode}.jpg"


def views_around(count: int, phi: float = POLAR_DEGREES) -> list[View]:
    return [View(n, round(360.0 * n / count, 2), phi) for n in range(count)]
