"""Small, non-financial helpers shared by presentation-only appendix notebooks."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def markdown_lines(lines: Iterable[str]) -> str:
    """Join Markdown lines using explicit HTML-compatible hard breaks."""
    return "  \n".join(str(line) for line in lines)


def draw_reference_rays(
    axis: Any,
    ratios: Iterable[float],
    *,
    x_max: float,
    y_max: float,
    color: str,
) -> None:
    """Draw and label constant-ratio rays without clipping labels outside an axis."""
    for ratio in ratios:
        x_end = min(float(x_max), float(y_max) / float(ratio))
        x = np.linspace(0.0, x_end, 100)
        axis.plot(x, float(ratio) * x, color=color, linewidth=0.65, linestyle=(0, (2, 2)), zorder=0)
        label_x = 0.96 * x_end
        axis.annotate(
            f"{ratio:.1f}×",
            (label_x, float(ratio) * label_x),
            xytext=(-2, 2),
            textcoords="offset points",
            ha="right",
            fontsize=5.8,
            color=color,
        )
