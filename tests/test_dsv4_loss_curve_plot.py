from __future__ import annotations

import csv
from pathlib import Path

from experiments.dsv4_muon_report_aligned.plot_curves import load_curve, render_svg


def test_loss_curve_svg_contains_both_projects_and_difference_panel(tmp_path: Path) -> None:
    csv_path = tmp_path / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cumulative_tokens", "total_loss", "comparison_total_loss"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"cumulative_tokens": 100, "total_loss": 4.0, "comparison_total_loss": 4.2},
                {"cumulative_tokens": 200, "total_loss": 3.5, "comparison_total_loss": 3.6},
                {"cumulative_tokens": 300, "total_loss": 3.1, "comparison_total_loss": 3.3},
            ]
        )

    curve = load_curve("B", csv_path, metric="total_loss")
    svg = render_svg(
        [curve],
        metric="total_loss",
        smooth_window=1,
        max_points=100,
        title="Loss comparison",
    )

    assert svg.startswith("<svg")
    assert "B TorchForge" in svg
    assert "B peer" in svg
    assert "absolute difference" in svg
    assert "nan" not in svg.lower()


def test_multi_series_svg_facets_overlapping_curves(tmp_path: Path) -> None:
    curves = []
    for label, offset in (("A", 0.0), ("B", 0.1), ("C", 0.1)):
        csv_path = tmp_path / f"{label}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["cumulative_tokens", "total_loss", "comparison_total_loss"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "cumulative_tokens": tokens,
                        "total_loss": loss + offset,
                        "comparison_total_loss": loss + offset + 1.0e-4,
                    }
                    for tokens, loss in ((100, 4.0), (200, 3.5), (300, 3.1))
                ]
            )
        curves.append(load_curve(label, csv_path, metric="total_loss"))

    svg = render_svg(
        curves,
        metric="total_loss",
        smooth_window=1,
        max_points=100,
        title="Loss comparison",
    )

    for label in ("A", "B", "C"):
        assert f"{label} total_loss" in svg
        assert f"{label} TorchForge" in svg
        assert f"{label} peer" in svg
        assert f"{label} |TorchForge-peer|" in svg
    assert 'stroke="#273444"' in svg
