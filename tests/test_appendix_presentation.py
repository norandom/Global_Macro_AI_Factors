from __future__ import annotations

import matplotlib.pyplot as plt

from macro_framework.appendix_presentation import draw_reference_rays, markdown_lines


def test_markdown_lines_uses_explicit_hard_breaks() -> None:
    assert markdown_lines(["first", "second"]) == "first  \nsecond"


def test_reference_ray_labels_remain_inside_supplied_bounds() -> None:
    figure, axis = plt.subplots()
    draw_reference_rays(axis, (0.8, 1.4), x_max=1.0, y_max=1.0, color="#000000")

    assert len(axis.lines) == 2
    assert all(text.xy[1] <= 1.0 for text in axis.texts)
    plt.close(figure)
