from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Result:
    phiAse: Sequence[float] = field(default_factory=list)
    standardError: Sequence[float] = field(default_factory=list)
    relativeStandardError: Sequence[float] = field(default_factory=list)
    totalRays: Sequence[int] = field(default_factory=list)
    dndtAse: Sequence[float] = field(default_factory=list)
    boundaryStatus: str = "disabled"
    boundaryPasses: int = 0
    boundaryRemainingFraction: float = 0.0
    boundaryMaxPasses: int = 0
    boundaryDivergenceStreak: int = 0
    boundaryGamma: float = 0.0
    boundaryGammaStandardError: float = 0.0
    boundaryTailFactor: float = 0.0
    boundaryTailClosure: float = 0.0
