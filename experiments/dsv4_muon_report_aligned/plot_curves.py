from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


COLORS = ("#146C94", "#C44536", "#2A9D55", "#7A5195", "#B56A00", "#4F5D75")


@dataclass(frozen=True)
class Curve:
    label: str
    torchforge: list[tuple[int, float]]
    comparison: list[tuple[int, float]]
    difference: list[tuple[int, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paired loss curves and their absolute differences as SVG.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Label and loss_curve_comparison.csv path; pass once for each A/B/C run.",
    )
    parser.add_argument(
        "--metric",
        default="total_loss",
        choices=("total_loss", "lm_loss", "mtp_loss", "aux_loss", "validation_loss"),
    )
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="397M cross-project loss comparison")
    parser.add_argument(
        "--optimizer-analysis",
        action="store_true",
        help="Add TorchForge/peer optimizer comparisons and signed A-B/A-C/B-C loss differences.",
    )
    return parser.parse_args()


def _number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Curve value must be finite, got {value!r}.")
    return result


def load_curve(label: str, path: str | Path, *, metric: str) -> Curve:
    path = Path(path)
    local_key = metric
    comparison_key = f"comparison_{metric}"
    torchforge = []
    comparison = []
    difference = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"cumulative_tokens", local_key, comparison_key}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            tokens_value = _number(row.get("cumulative_tokens"))
            if tokens_value is None or int(tokens_value) != tokens_value:
                raise ValueError(f"Invalid cumulative_tokens in {path}: {row.get('cumulative_tokens')!r}")
            tokens = int(tokens_value)
            local = _number(row.get(local_key))
            peer = _number(row.get(comparison_key))
            if local is not None:
                torchforge.append((tokens, local))
            if peer is not None:
                comparison.append((tokens, peer))
            if local is not None and peer is not None:
                difference.append((tokens, abs(local - peer)))
    if not torchforge or not comparison:
        raise ValueError(f"{path} does not contain paired values for metric {metric}.")
    return Curve(label=label, torchforge=torchforge, comparison=comparison, difference=difference)


def smooth(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 0:
        raise ValueError("smooth_window must be positive.")
    if window == 1:
        return list(points)
    result = []
    values: list[float] = []
    running = 0.0
    for tokens, value in points:
        values.append(value)
        running += value
        if len(values) > window:
            running -= values[-window - 1]
        result.append((tokens, running / min(len(values), window)))
    return result


def decimate(points: list[tuple[int, float]], max_points: int) -> list[tuple[int, float]]:
    if max_points < 2:
        raise ValueError("max_points must be at least 2.")
    if len(points) <= max_points:
        return points
    stride = math.ceil((len(points) - 1) / (max_points - 1))
    result = points[::stride]
    if result[-1] != points[-1]:
        result.append(points[-1])
    return result


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        return [low]
    return [low + index * (high - low) / count for index in range(count + 1)]


def _path(
    points: Iterable[tuple[int, float]],
    *,
    x_min: int,
    x_max: int,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    commands = []
    for index, (tokens, value) in enumerate(points):
        x = left + (tokens - x_min) / max(x_max - x_min, 1) * width
        y = top + height - (value - y_min) / max(y_max - y_min, 1.0e-30) * height
        commands.append(f"{'M' if index == 0 else 'L'}{x:.2f},{y:.2f}")
    return " ".join(commands)


def _signed_difference(
    left_points: list[tuple[int, float]],
    right_points: list[tuple[int, float]],
    *,
    label: str,
) -> list[tuple[int, float]]:
    left = dict(left_points)
    right = dict(right_points)
    if left.keys() != right.keys():
        raise ValueError(f"Optimizer comparison {label} requires identical cumulative-token positions.")
    return [(tokens, left[tokens] - right[tokens]) for tokens in sorted(left)]


def render_svg(
    curves: list[Curve],
    *,
    metric: str,
    smooth_window: int,
    max_points: int,
    title: str,
    optimizer_analysis: bool = False,
) -> str:
    prepared = []
    for curve in curves:
        prepared.append(
            Curve(
                label=curve.label,
                torchforge=decimate(smooth(curve.torchforge, smooth_window), max_points),
                comparison=decimate(smooth(curve.comparison, smooth_window), max_points),
                difference=decimate(smooth(curve.difference, smooth_window), max_points),
            )
        )
    all_loss = [point for curve in prepared for points in (curve.torchforge, curve.comparison) for point in points]
    all_diff = [point for curve in prepared for point in curve.difference]
    if not all_loss or not all_diff:
        raise ValueError("No paired curve points are available to plot.")
    x_min = min(tokens for tokens, _ in all_loss)
    x_max = max(tokens for tokens, _ in all_loss)
    loss_min = min(value for _, value in all_loss)
    loss_max = max(value for _, value in all_loss)
    loss_padding = max((loss_max - loss_min) * 0.08, 1.0e-6)
    loss_min -= loss_padding
    loss_max += loss_padding
    diff_max = max(value for _, value in all_diff)
    diff_max = max(diff_max * 1.1, 1.0e-12)

    pairwise_torchforge: list[tuple[str, list[tuple[int, float]]]] = []
    pairwise_peer: list[tuple[str, list[tuple[int, float]]]] = []
    if optimizer_analysis:
        if len(prepared) < 2:
            raise ValueError("--optimizer-analysis requires at least two series.")
        for left_index, left_curve in enumerate(prepared):
            for right_curve in prepared[left_index + 1 :]:
                pair_label = f"{left_curve.label}-{right_curve.label}"
                pairwise_torchforge.append(
                    (
                        pair_label,
                        _signed_difference(
                            left_curve.torchforge,
                            right_curve.torchforge,
                            label=f"TorchForge {pair_label}",
                        ),
                    )
                )
                pairwise_peer.append(
                    (
                        pair_label,
                        _signed_difference(
                            left_curve.comparison,
                            right_curve.comparison,
                            label=f"peer {pair_label}",
                        ),
                    )
                )
        signed_values = [
            abs(value)
            for _, points in (*pairwise_torchforge, *pairwise_peer)
            for _, value in points
        ]
        signed_max = max(max(signed_values, default=0.0) * 1.1, 1.0e-12)
    else:
        signed_max = 1.0

    canvas_width = 1200
    left, right = 92.0, 32.0
    plot_width = canvas_width - left - right
    loss_height, loss_gap = 180.0, 34.0
    loss_tops = [112.0 + index * (loss_height + loss_gap) for index in range(len(prepared))]
    diff_height, diff_gap = 120.0, 34.0
    diff_tops = [
        loss_tops[-1] + loss_height + 70.0 + index * (diff_height + diff_gap)
        for index in range(len(prepared))
    ]
    analysis_tops: list[float] = []
    analysis_heights: list[float] = []
    cursor = diff_tops[-1] + diff_height
    if optimizer_analysis:
        cursor += 70.0
        for height in (180.0, 180.0, 150.0, 150.0):
            analysis_tops.append(cursor)
            analysis_heights.append(height)
            cursor += height + 34.0
        cursor -= 34.0
    final_panel_top = analysis_tops[-1] if analysis_tops else diff_tops[-1]
    final_panel_height = analysis_heights[-1] if analysis_heights else diff_height
    canvas_height = int(final_panel_top + final_panel_height + 70.0)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#17212B">{escape(title)}</text>',
        f'<text x="{left}" y="70" font-family="Arial, sans-serif" font-size="14" fill="#52606D">Metric: {escape(metric)}; trailing smoothing window: {smooth_window}; exact cumulative-token alignment</text>',
    ]

    def grid(top: float, height: float, y_min: float, y_max: float, label: str) -> None:
        elements.append(f'<text x="18" y="{top + height / 2:.1f}" transform="rotate(-90 18 {top + height / 2:.1f})" font-family="Arial, sans-serif" font-size="13" fill="#3E4C59">{escape(label)}</text>')
        for tick in _ticks(y_min, y_max):
            y = top + height - (tick - y_min) / max(y_max - y_min, 1.0e-30) * height
            elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#E4E7EB" stroke-width="1"/>')
            elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#66788A">{tick:.4g}</text>')
        elements.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{height}" fill="none" stroke="#9AA5B1" stroke-width="1"/>')

    panel_boxes = [
        *((top, loss_height) for top in loss_tops),
        *((top, diff_height) for top in diff_tops),
        *zip(analysis_tops, analysis_heights),
    ]
    for top, height in panel_boxes:
        for tick in _ticks(float(x_min), float(x_max)):
            x = left + (tick - x_min) / max(x_max - x_min, 1) * plot_width
            elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + height}" stroke="#EEF1F4" stroke-width="1"/>')

    for top in loss_tops:
        grid(top, loss_height, loss_min, loss_max, metric)
    for top in diff_tops:
        grid(top, diff_height, 0.0, diff_max, "absolute difference")
    if optimizer_analysis:
        grid(analysis_tops[0], analysis_heights[0], loss_min, loss_max, metric)
        grid(analysis_tops[1], analysis_heights[1], loss_min, loss_max, metric)
        grid(analysis_tops[2], analysis_heights[2], -signed_max, signed_max, "signed loss difference")
        grid(analysis_tops[3], analysis_heights[3], -signed_max, signed_max, "signed loss difference")
    for tick in _ticks(float(x_min), float(x_max)):
        x = left + (tick - x_min) / max(x_max - x_min, 1) * plot_width
        elements.append(f'<text x="{x:.2f}" y="{final_panel_top + final_panel_height + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#66788A">{tick / 1.0e9:.3g}B</text>')

    for index, (curve, loss_top, diff_top) in enumerate(zip(prepared, loss_tops, diff_tops)):
        color = COLORS[index % len(COLORS)]
        local_path = _path(curve.torchforge, x_min=x_min, x_max=x_max, y_min=loss_min, y_max=loss_max, left=left, top=loss_top, width=plot_width, height=loss_height)
        peer_path = _path(curve.comparison, x_min=x_min, x_max=x_max, y_min=loss_min, y_max=loss_max, left=left, top=loss_top, width=plot_width, height=loss_height)
        diff_path = _path(curve.difference, x_min=x_min, x_max=x_max, y_min=0.0, y_max=diff_max, left=left, top=diff_top, width=plot_width, height=diff_height)
        elements.append(f'<path d="{local_path}" fill="none" stroke="{color}" stroke-width="2.8"/>')
        elements.append(f'<path d="{peer_path}" fill="none" stroke="#273444" stroke-width="2.0" stroke-dasharray="8 5" opacity="0.95"/>')
        elements.append(f'<path d="{diff_path}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        elements.append(f'<text x="{left + 10}" y="{loss_top + 22}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#17212B">{escape(curve.label)} {escape(metric)}</text>')
        legend_x = left + plot_width - 260
        legend_y = loss_top + 18
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="2.8"/>')
        elements.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#323F4B">{escape(curve.label)} TorchForge</text>')
        legend_x += 140
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="#273444" stroke-width="2" stroke-dasharray="8 5"/>')
        elements.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#323F4B">{escape(curve.label)} peer</text>')
        elements.append(f'<text x="{left + 10}" y="{diff_top + 22}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#17212B">{escape(curve.label)} absolute TorchForge-peer difference</text>')
        elements.append(f'<line x1="{left + plot_width - 190}" y1="{diff_top + 18}" x2="{left + plot_width - 166}" y2="{diff_top + 18}" stroke="{color}" stroke-width="2"/>')
        elements.append(f'<text x="{left + plot_width - 160}" y="{diff_top + 22}" font-family="Arial, sans-serif" font-size="12" fill="#323F4B">{escape(curve.label)} |TorchForge-peer|</text>')

    if optimizer_analysis:
        analysis_specs = (
            ("TorchForge A/B/C optimizer comparison", "torchforge", analysis_tops[0], analysis_heights[0]),
            ("peer A/B/C optimizer comparison", "comparison", analysis_tops[1], analysis_heights[1]),
        )
        for panel_title, source, top, height in analysis_specs:
            legend_x = left + 320
            for index, curve in enumerate(prepared):
                color = COLORS[index % len(COLORS)]
                points = curve.torchforge if source == "torchforge" else curve.comparison
                path = _path(points, x_min=x_min, x_max=x_max, y_min=loss_min, y_max=loss_max, left=left, top=top, width=plot_width, height=height)
                dash = "" if source == "torchforge" else ' stroke-dasharray="8 5"'
                elements.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
                elements.append(f'<line x1="{legend_x}" y1="{top + 18}" x2="{legend_x + 24}" y2="{top + 18}" stroke="{color}" stroke-width="2.2"{dash}/>')
                elements.append(f'<text x="{legend_x + 30}" y="{top + 22}" font-family="Arial, sans-serif" font-size="12" fill="#323F4B">{escape(curve.label)}</text>')
                legend_x += 100
            elements.append(f'<text x="{left + 10}" y="{top + 22}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#17212B">{escape(panel_title)}</text>')

        pair_specs = (
            ("TorchForge signed optimizer differences (left minus right)", pairwise_torchforge, analysis_tops[2], analysis_heights[2]),
            ("peer signed optimizer differences (left minus right)", pairwise_peer, analysis_tops[3], analysis_heights[3]),
        )
        for panel_title, pairs, top, height in pair_specs:
            zero_y = top + height / 2
            elements.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_width}" y2="{zero_y:.2f}" stroke="#66788A" stroke-width="1.2"/>')
            legend_x = left + 430
            for index, (pair_label, points) in enumerate(pairs):
                color = COLORS[(index + 3) % len(COLORS)]
                path = _path(points, x_min=x_min, x_max=x_max, y_min=-signed_max, y_max=signed_max, left=left, top=top, width=plot_width, height=height)
                elements.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
                elements.append(f'<line x1="{legend_x}" y1="{top + 18}" x2="{legend_x + 24}" y2="{top + 18}" stroke="{color}" stroke-width="2"/>')
                elements.append(f'<text x="{legend_x + 30}" y="{top + 22}" font-family="Arial, sans-serif" font-size="12" fill="#323F4B">{escape(pair_label)}</text>')
                legend_x += 110
            elements.append(f'<text x="{left + 10}" y="{top + 22}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#17212B">{escape(panel_title)}</text>')
    elements.append(f'<text x="{left + plot_width / 2:.1f}" y="{canvas_height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#3E4C59">Cumulative tokens (billions)</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def _series_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=CSV for --series, got {value!r}.")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError(f"Expected non-empty LABEL=CSV for --series, got {value!r}.")
    return label, Path(path)


def main() -> int:
    args = parse_args()
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be positive.")
    curves = [load_curve(label, path, metric=args.metric) for label, path in map(_series_argument, args.series)]
    svg = render_svg(
        curves,
        metric=args.metric,
        smooth_window=args.smooth_window,
        max_points=args.max_points,
        title=args.title,
        optimizer_analysis=args.optimizer_analysis,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
