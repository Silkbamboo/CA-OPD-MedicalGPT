#!/usr/bin/env python3
"""Render the README figures from public aggregate experiment artifacts.

The script deliberately uses only the Python standard library.  Result values are
read from ``artifacts/results``; no checkpoint score is embedded in the drawing
code.  The generated SVG files are deterministic and can be checked in CI with
``--check``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "artifacts" / "results"
DEFAULT_ROUTER_CONFIG = ROOT / "configs" / "public" / "ca_opd_stage120.recorded.json"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "readme"

INK = "#0f172a"
MUTED = "#475569"
LIGHT = "#f8fafc"
PANEL = "#ffffff"
BORDER = "#cbd5e1"
GRID = "#e2e8f0"
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#059669"
PURPLE = "#7c3aed"
RED = "#dc2626"
PALE_BLUE = "#dbeafe"
PALE_ORANGE = "#ffedd5"
PALE_GREEN = "#dcfce7"
PALE_PURPLE = "#ede9fe"
PALE_RED = "#fee2e2"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def verify_result_hashes(results_dir: Path) -> None:
    checksum_path = results_dir / "SHA256SUMS"
    for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        expected, name = raw_line.split(maxsplit=1)
        actual = hashlib.sha256((results_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {expected}")


def pct(correct: int, total: int) -> float:
    return 100.0 * correct / total


def scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high <= low:
        raise ValueError("scale domain must be increasing")
    return start + (value - low) * (end - start) / (high - low)


def fmt_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def attrs(**values: object) -> str:
    rendered: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        name = key.removesuffix("_").replace("_", "-")
        rendered.append(f'{name}="{html.escape(str(value), quote=True)}"')
    return " ".join(rendered)


def text(
    x: float,
    y: float,
    value: object,
    *,
    class_: str = "label",
    anchor: str = "start",
    fill: str | None = None,
) -> str:
    return f"<text {attrs(x=round(x, 2), y=round(y, 2), class_=class_, text_anchor=anchor, fill=fill)}>" f"{html.escape(str(value))}</text>"


def multiline(
    x: float,
    y: float,
    lines: Iterable[object],
    *,
    class_: str = "box-copy",
    anchor: str = "start",
    gap: float = 23,
) -> str:
    return "".join(text(x, y + index * gap, line_value, class_=class_, anchor=anchor) for index, line_value in enumerate(lines))


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = PANEL,
    stroke: str = BORDER,
    radius: float = 12,
    class_: str | None = None,
    dash: str | None = None,
) -> str:
    return f"<rect {attrs(x=x, y=y, width=width, height=height, rx=radius, fill=fill, stroke=stroke, class_=class_, stroke_dasharray=dash)}/>"


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = BORDER,
    width: float = 1.5,
    dash: str | None = None,
    marker_end: str | None = None,
    class_: str | None = None,
) -> str:
    return f"<line {attrs(x1=round(x1, 2), y1=round(y1, 2), x2=round(x2, 2), y2=round(y2, 2), stroke=stroke, stroke_width=width, stroke_dasharray=dash, marker_end=marker_end, class_=class_)}/>"


def path(
    d: str,
    *,
    stroke: str = MUTED,
    width: float = 2,
    dash: str | None = None,
    marker_end: str | None = None,
) -> str:
    return f"<path {attrs(d=d, fill='none', stroke=stroke, stroke_width=width, stroke_dasharray=dash, marker_end=marker_end)}/>"


def circle(
    cx: float,
    cy: float,
    radius: float,
    *,
    fill: str,
    stroke: str = PANEL,
    width: float = 2,
) -> str:
    return f"<circle {attrs(cx=round(cx, 2), cy=round(cy, 2), r=radius, fill=fill, stroke=stroke, stroke_width=width)}/>"


def box(
    body: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    title_value: str,
    copy_lines: Iterable[str],
    *,
    fill: str = PANEL,
    stroke: str = BORDER,
    dashed: bool = False,
) -> None:
    body.append(
        rect(
            x,
            y,
            width,
            height,
            fill=fill,
            stroke=stroke,
            dash="8 6" if dashed else None,
        )
    )
    body.append(text(x + 18, y + 32, title_value, class_="box-title"))
    body.append(multiline(x + 18, y + 61, copy_lines, gap=22))


def document(title_value: str, description: str, body: Iterable[str], width: int, height: int) -> str:
    style = f"""
    text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans SC',
           'Microsoft YaHei', Arial, sans-serif; fill: {INK}; }}
    .figure-title {{ font-size: 28px; font-weight: 700; }}
    .figure-subtitle {{ font-size: 16px; fill: {MUTED}; }}
    .panel-title {{ font-size: 19px; font-weight: 700; }}
    .box-title {{ font-size: 18px; font-weight: 700; }}
    .box-copy {{ font-size: 16px; fill: {MUTED}; }}
    .label {{ font-size: 16px; }}
    .small {{ font-size: 15px; fill: {MUTED}; }}
    .axis {{ font-size: 15px; fill: {MUTED}; }}
    .axis-title {{ font-size: 16px; font-weight: 600; fill: {INK}; }}
    .value {{ font-size: 16px; font-weight: 700; }}
    .note {{ font-size: 15px; fill: {MUTED}; }}
    .flow {{ stroke-linecap: round; stroke-linejoin: round; }}
    """
    content = "\n  ".join(body)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title_value)}</title>
  <desc id="desc">{html.escape(description)}</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/>
    </marker>
  </defs>
  <style>{style}</style>
  <rect width="{width}" height="{height}" fill="{PANEL}"/>
  {content}
</svg>
"""


def add_x_axis(
    body: list[str],
    left: float,
    right: float,
    y: float,
    domain: tuple[float, float],
    ticks: Iterable[float],
    label_value: str,
) -> Callable[[float], float]:
    def project(value: float) -> float:
        return scale(value, domain[0], domain[1], left, right)

    body.append(line(left, y, right, y, stroke=MUTED, width=1.3))
    for tick in ticks:
        x = project(tick)
        body.append(line(x, y, x, y + 7, stroke=MUTED, width=1.2))
        body.append(text(x, y + 27, f"{tick:g}", class_="axis", anchor="middle"))
    body.append(
        text(
            (left + right) / 2,
            y + 53,
            label_value,
            class_="axis-title",
            anchor="middle",
        )
    )
    return project


def add_y_axis(
    body: list[str],
    x: float,
    top: float,
    bottom: float,
    domain: tuple[float, float],
    ticks: Iterable[float],
    label_value: str,
) -> Callable[[float], float]:
    def project(value: float) -> float:
        return scale(value, domain[0], domain[1], bottom, top)

    body.append(line(x, top, x, bottom, stroke=MUTED, width=1.3))
    for tick in ticks:
        y = project(tick)
        body.append(line(x, y, x - 7, y, stroke=MUTED, width=1.2))
        body.append(line(x, y, x + 1, y, stroke=GRID, width=1))
        body.append(text(x - 12, y + 5, f"{tick:g}", class_="axis", anchor="end"))
    body.append(f'<text {attrs(x=x - 58, y=(top + bottom) / 2, class_="axis-title", text_anchor="middle", transform=f"rotate(-90 {x - 58} {(top + bottom) / 2})")}>{html.escape(label_value)}</text>')
    return project


def render_opd_training_loop() -> str:
    body: list[str] = []
    body.append(
        text(
            600,
            45,
            "Same-trajectory OPD: four probability identities",
            class_="figure-title",
            anchor="middle",
        )
    )
    body.append(
        text(
            600,
            73,
            "B2, IDT and CA share the scoring, PPO and transaction skeleton; routing and safety controls are protocol-specific.",
            class_="figure-subtitle",
            anchor="middle",
        )
    )

    box(
        body,
        35,
        150,
        165,
        150,
        "Prompt batch",
        ["No answer fields", "Medical or General"],
        fill=LIGHT,
    )
    box(
        body,
        235,
        150,
        175,
        150,
        "Student rollout",
        ["behavior policy μ", "completion tokens"],
        fill=PALE_BLUE,
        stroke=BLUE,
    )
    box(
        body,
        445,
        150,
        170,
        150,
        "Frozen trajectory",
        ["same prompt", "same token path"],
        fill=LIGHT,
    )

    body.append(rect(650, 100, 315, 250, fill=PANEL, stroke=PURPLE))
    body.append(text(668, 133, "Token probabilities", class_="box-title"))
    probability_rows = [
        (BLUE, "μ", "rollout logprob · detached"),
        (MUTED, "p_old", "old actor · detached"),
        (PURPLE, "πθ", "current actor · gradient"),
        (ORANGE, "πT", "selected Teacher · detached"),
    ]
    for index, (colour, symbol, meaning) in enumerate(probability_rows):
        y = 172 + index * 43
        body.append(circle(675, y - 5, 6, fill=colour, stroke=colour, width=1))
        body.append(text(694, y, symbol, class_="value", fill=colour))
        body.append(text(742, y, meaning, class_="box-copy"))

    box(
        body,
        1000,
        150,
        165,
        150,
        "PPO candidate",
        ["advantage A_t", "PPO ratio r_t", "token correction c_t", "prompt-equal mean"],
        fill=PALE_PURPLE,
        stroke=PURPLE,
    )
    box(
        body,
        1000,
        405,
        165,
        130,
        "Transaction gate",
        [
            "identity · finite · ESS",
            "grad · generation",
            "commit or rollback",
        ],
        fill=PALE_GREEN,
        stroke=GREEN,
    )

    for start, end in [
        ((200, 225), (235, 225)),
        ((410, 225), (445, 225)),
        ((615, 225), (650, 225)),
        ((965, 225), (1000, 225)),
    ]:
        body.append(
            line(
                *start,
                *end,
                stroke=MUTED,
                width=2.2,
                marker_end="url(#arrow)",
                class_="flow",
            )
        )
    body.append(
        line(
            1082,
            300,
            1082,
            405,
            stroke=MUTED,
            width=2.2,
            marker_end="url(#arrow)",
            class_="flow",
        )
    )
    body.append(path("M 1000 482 H 322 V 310", stroke=GREEN, width=2.2, marker_end="url(#arrow)"))
    body.append(
        text(
            625,
            505,
            "commit LoRA / optimizer / RNG / cursor; refresh and verify sampler",
            class_="note",
            anchor="middle",
            fill=GREEN,
        )
    )
    body.append(
        text(
            1082,
            378,
            "health checks before commit",
            class_="note",
            anchor="middle",
        )
    )
    body.append(
        text(
            600,
            585,
            "Teacher never generates a replacement answer; all four probabilities are evaluated on the Student's sampled tokens.",
            class_="figure-subtitle",
            anchor="middle",
        )
    )
    return document(
        "Same-trajectory OPD training loop",
        "Prompt-only data feed a Student rollout. Behavior, old actor, current actor and selected Teacher probabilities are kept distinct before a transactional PPO-style update.",
        body,
        1200,
        620,
    )


def render_ca_routing() -> str:
    body: list[str] = []
    body.append(
        text(
            600,
            45,
            "CA-OPD: constraint-aware Teacher routing",
            class_="figure-title",
            anchor="middle",
        )
    )
    body.append(
        text(
            600,
            73,
            "Controller feedback changes the next training window; confirmation and final labels remain outside the training graph.",
            class_="figure-subtitle",
            anchor="middle",
        )
    )

    box(
        body,
        35,
        145,
        185,
        160,
        "Controller dev",
        ["Medical accuracy M_k", "General accuracy G_k", "label-isolated evaluator"],
        fill=PALE_GREEN,
        stroke=GREEN,
    )
    box(
        body,
        260,
        145,
        180,
        160,
        "Capability gaps",
        ["EMA smoothing", "medical target gap", "general floor gap"],
        fill=LIGHT,
    )
    box(
        body,
        480,
        125,
        215,
        200,
        "Router state",
        ["softmax + [p_min, p_max]", "hysteresis", "RECOVER_GENERAL:", "p_M = p_min"],
        fill=PALE_PURPLE,
        stroke=PURPLE,
    )
    box(
        body,
        740,
        95,
        190,
        135,
        "Medical route",
        ["Medical prompt pool", "B1 SFT Teacher", "probability p_M"],
        fill=PALE_ORANGE,
        stroke=ORANGE,
    )
    box(
        body,
        740,
        270,
        190,
        135,
        "General route",
        ["General anchors", "B0 Base Teacher", "probability 1 - p_M"],
        fill=PALE_BLUE,
        stroke=BLUE,
    )
    box(
        body,
        975,
        175,
        190,
        170,
        "Shared Student update",
        [
            "Student rollout",
            "same-trajectory scoring",
            "correction + PPO",
            "transaction gate",
        ],
        fill=LIGHT,
        stroke=INK,
    )
    box(
        body,
        35,
        385,
        405,
        105,
        "Capability boundary",
        [
            "Controller may feed the router.",
            "Confirmation / final cannot be imported by trainer or router.",
        ],
        fill=PALE_RED,
        stroke=RED,
        dashed=True,
    )

    body.append(line(220, 225, 260, 225, stroke=MUTED, width=2.2, marker_end="url(#arrow)"))
    body.append(line(440, 225, 480, 225, stroke=MUTED, width=2.2, marker_end="url(#arrow)"))
    body.append(path("M 695 205 C 715 205 720 163 740 163", width=2.2, marker_end="url(#arrow)"))
    body.append(path("M 695 245 C 715 245 720 338 740 338", width=2.2, marker_end="url(#arrow)"))
    body.append(
        path(
            "M 930 163 C 950 163 950 225 975 225",
            stroke=ORANGE,
            width=2.2,
            marker_end="url(#arrow)",
        )
    )
    body.append(
        path(
            "M 930 338 C 950 338 950 295 975 295",
            stroke=BLUE,
            width=2.2,
            marker_end="url(#arrow)",
        )
    )
    body.append(
        path(
            "M 1070 345 V 530 H 128 V 305",
            stroke=GREEN,
            width=2.2,
            marker_end="url(#arrow)",
        )
    )
    body.append(
        text(
            602,
            553,
            "evaluate every 30 accepted steps → update the next route window",
            class_="note",
            anchor="middle",
            fill=GREEN,
        )
    )
    return document(
        "Constraint-aware CA-OPD routing loop",
        "Medical and General Controller scores are smoothed into capability gaps. A bounded, hysteretic router selects a Medical or Base Teacher route for the shared Student update, then evaluates again every thirty accepted steps.",
        body,
        1200,
        600,
    )


def forest_row(
    body: list[str],
    project: Callable[[float], float],
    y: float,
    label_value: str,
    estimate: float,
    low: float,
    high: float,
    p_value: float,
    colour: str,
    *,
    label_x: float = 55,
) -> None:
    body.append(text(label_x, y - 17, label_value, class_="label"))
    body.append(text(label_x, y + 9, f"McNemar p = {p_value:.4g}", class_="small"))
    body.append(line(project(low), y, project(high), y, stroke=colour, width=5))
    body.append(line(project(low), y - 9, project(low), y + 9, stroke=colour, width=2))
    body.append(line(project(high), y - 9, project(high), y + 9, stroke=colour, width=2))
    body.append(circle(project(estimate), y, 8, fill=colour, stroke=PANEL, width=2))
    body.append(
        text(
            project(estimate),
            y - 18,
            f"{fmt_signed(estimate)} pp",
            class_="value",
            anchor="middle",
            fill=colour,
        )
    )
    body.append(
        text(
            project(estimate),
            y + 26,
            f"95% CI [{fmt_signed(low)}, {fmt_signed(high)}]",
            class_="small",
            anchor="middle",
        )
    )


def render_experiment_overview(data: dict) -> str:
    sft = data["sft"]
    p10 = data["p10"]
    stage = data["stage"]
    general_floor = data["general_floor"]

    body: list[str] = []
    body.append(
        text(
            600,
            42,
            "Key results: paired confirmation and capability plane",
            class_="figure-title",
            anchor="middle",
        )
    )
    body.append(
        text(
            600,
            70,
            "Development / confirmation protocols only · single seed · final-test access = 0",
            class_="figure-subtitle",
            anchor="middle",
        )
    )
    body.append(rect(25, 100, 500, 470, fill=LIGHT, stroke=BORDER, radius=14))
    body.append(rect(550, 100, 625, 470, fill=LIGHT, stroke=BORDER, radius=14))

    body.append(text(50, 135, "A · Paired accuracy difference vs Base", class_="panel-title"))
    body.append(
        text(
            50,
            161,
            "600-question confirmation · paired bootstrap 95% CI",
            class_="small",
        )
    )
    forest_left, forest_right, forest_axis_y = 180, 495, 495
    project_effect = add_x_axis(
        body,
        forest_left,
        forest_right,
        forest_axis_y,
        (-2.0, 8.0),
        [-2, 0, 2, 4, 6, 8],
        "Difference vs Base (percentage points)",
    )
    body.append(
        line(
            project_effect(0),
            190,
            project_effect(0),
            forest_axis_y,
            stroke=MUTED,
            width=1.5,
            dash="6 5",
        )
    )

    sft_ci = [100.0 * value for value in sft["paired_bootstrap_95_ci"]]
    p10_ci = list(map(float, p10["paired_bootstrap_95_ci_percentage_points"]))
    forest_row(
        body,
        project_effect,
        270,
        "B1 SFT-v3 step450 - B0",
        float(sft["delta_percentage_points"]),
        sft_ci[0],
        sft_ci[1],
        float(sft["mcnemar_exact_two_sided_p"]),
        GREEN,
    )
    forest_row(
        body,
        project_effect,
        395,
        "B2 step240 - B0",
        float(p10["delta_percentage_points"]),
        p10_ci[0],
        p10_ci[1],
        float(p10["mcnemar_exact_two_sided_p"]),
        ORANGE,
    )

    body.append(text(575, 135, "B · Medical-General outcome plane", class_="panel-title"))
    body.append(text(575, 161, "Controller dev: Medical n=300, General n=209", class_="small"))
    plot_left, plot_right, plot_top, plot_bottom = 650, 1140, 195, 485
    project_general = add_x_axis(
        body,
        plot_left,
        plot_right,
        plot_bottom,
        (59.5, 67.2),
        [60, 62, 64, 66],
        "General accuracy (%)",
    )
    project_medical = add_y_axis(
        body,
        plot_left,
        plot_top,
        plot_bottom,
        (71.0, 80.5),
        [72, 74, 76, 78, 80],
        "Medical accuracy (%)",
    )
    floor_x = project_general(general_floor)
    body.append(f"<rect {attrs(x=plot_left, y=plot_top, width=max(0, floor_x - plot_left), height=plot_bottom - plot_top, fill=PALE_RED, opacity=0.65)}/>")
    body.append(line(floor_x, plot_top, floor_x, plot_bottom, stroke=RED, width=1.8, dash="7 5"))
    body.append(
        text(
            floor_x + 7,
            plot_top + 18,
            f"General floor {general_floor:.3f}%",
            class_="small",
            fill=RED,
        )
    )

    route_names = {
        "B0": "B0 Base",
        "B1_sft_v3": "B1 SFT-v3",
        "B2_step120": "B2",
        "IDT_step120": "IDT",
        "CA_step120": "CA",
    }
    route_colours = {
        "B0": INK,
        "B1_sft_v3": GREEN,
        "B2_step120": BLUE,
        "IDT_step120": PURPLE,
        "CA_step120": ORANGE,
    }
    label_offsets = {
        "B0": (12, -14),
        "B1_sft_v3": (-12, -12),
        "B2_step120": (12, 20),
        "IDT_step120": (-12, -24),
        "CA_step120": (-12, 20),
    }
    for route_key, values in stage["routes"].items():
        general = pct(values["general_correct"], stage["general_total"])
        medical = pct(values["medical_correct"], stage["medical_total"])
        px = project_general(general)
        py = project_medical(medical)
        colour = route_colours[route_key]
        dx, dy = label_offsets[route_key]
        body.append(circle(px, py, 8, fill=colour, stroke=PANEL, width=2))
        body.append(
            text(
                px + dx,
                py + dy,
                route_names[route_key],
                class_="value",
                anchor="end" if dx < 0 else "start",
                fill=colour,
            )
        )
    body.append(
        text(
            862,
            558,
            "B2 / IDT / CA: 120 × 4 prompts; B1: SFT reference (unequal budget).",
            class_="small",
            anchor="middle",
        )
    )
    return document(
        "Key experiment results",
        "A forest plot shows SFT-v3 and B2 paired differences against Base on a six-hundred-question development confirmation protocol. A capability plane shows Medical and General Controller outcomes with the frozen General constraint.",
        body,
        1200,
        600,
    )


def render_b2_dose_confirmation(data: dict) -> str:
    curve = data["curve"]
    p10 = data["p10"]
    body: list[str] = []
    body.append(
        text(
            600,
            42,
            "B2 dose-response: development selection versus isolated confirmation",
            class_="figure-title",
            anchor="middle",
        )
    )
    body.append(
        text(
            600,
            70,
            "Sparse accepted-checkpoint evaluations; connecting lines are visual guides, not unobserved measurements.",
            class_="figure-subtitle",
            anchor="middle",
        )
    )
    body.append(rect(25, 100, 735, 470, fill=LIGHT, stroke=BORDER, radius=14))
    body.append(rect(785, 100, 390, 470, fill=LIGHT, stroke=BORDER, radius=14))

    body.append(text(50, 135, "A · Controller delta from B0", class_="panel-title"))
    body.append(
        text(
            50,
            161,
            "Medical n=300 · General n=209 · accepted checkpoints only",
            class_="small",
        )
    )
    plot_left, plot_right, plot_top, plot_bottom = 115, 725, 195, 485
    project_step = add_x_axis(
        body,
        plot_left,
        plot_right,
        plot_bottom,
        (120, 300),
        [120, 150, 180, 210, 240, 270, 300],
        "Accepted step",
    )
    project_delta = add_y_axis(
        body,
        plot_left,
        plot_top,
        plot_bottom,
        (-3.0, 2.0),
        [-3, -2, -1, 0, 1, 2],
        "Accuracy delta vs B0 (pp)",
    )
    body.append(
        line(
            plot_left,
            project_delta(0),
            plot_right,
            project_delta(0),
            stroke=MUTED,
            width=1.5,
            dash="7 5",
        )
    )
    body.append(text(plot_left + 8, project_delta(0) - 8, "B0 reference", class_="small"))
    body.append(
        line(
            plot_left,
            project_delta(-1),
            plot_right,
            project_delta(-1),
            stroke=ORANGE,
            width=1.4,
            dash="4 5",
        )
    )
    body.append(
        text(
            plot_left + 8,
            project_delta(-1) - 8,
            "General floor (-1 pp)",
            class_="small",
            fill=ORANGE,
        )
    )

    selected_step = int(curve["selected_development_step"])
    selected_x = project_step(selected_step)
    body.append(
        line(
            selected_x,
            plot_top,
            selected_x,
            plot_bottom,
            stroke=PURPLE,
            width=1.5,
            dash="7 5",
        )
    )
    body.append(
        text(
            selected_x + 7,
            plot_top + 18,
            "dev-selected step240",
            class_="small",
            fill=PURPLE,
        )
    )

    series = {"Medical": (BLUE, []), "General": (ORANGE, [])}
    base_medical = pct(curve["base"]["medical_correct"], curve["medical_total"])
    base_general = pct(curve["base"]["general_correct"], curve["general_total"])
    for point in curve["points"]:
        step_value = float(point["step"])
        series["Medical"][1].append(
            (
                project_step(step_value),
                project_delta(pct(point["medical_correct"], curve["medical_total"]) - base_medical),
            )
        )
        series["General"][1].append(
            (
                project_step(step_value),
                project_delta(pct(point["general_correct"], curve["general_total"]) - base_general),
            )
        )
    for series_name, (colour, points) in series.items():
        point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        body.append(f'<polyline {attrs(points=point_string, fill="none", stroke=colour, stroke_width=3, stroke_linecap="round", stroke_linejoin="round")}/>')
        for x, y in points:
            is_selected = abs(x - selected_x) < 0.1
            body.append(
                circle(
                    x,
                    y,
                    8 if is_selected else 5.5,
                    fill=PANEL if is_selected else colour,
                    stroke=colour,
                    width=3 if is_selected else 2,
                )
            )
    body.append(line(505, 145, 535, 145, stroke=BLUE, width=3))
    body.append(circle(520, 145, 5, fill=BLUE, stroke=BLUE, width=1))
    body.append(text(545, 151, "Medical", class_="label", fill=BLUE))
    body.append(line(625, 145, 655, 145, stroke=ORANGE, width=3))
    body.append(circle(640, 145, 5, fill=ORANGE, stroke=ORANGE, width=1))
    body.append(text(665, 151, "General", class_="label", fill=ORANGE))

    body.append(text(810, 135, "B · Medical effect size", class_="panel-title"))
    body.append(text(810, 161, "Paired difference vs B0 with bootstrap 95% CI", class_="small"))
    effect_left, effect_right, effect_axis_y = 930, 1140, 495
    project_effect = add_x_axis(
        body,
        effect_left,
        effect_right,
        effect_axis_y,
        (-2.0, 5.0),
        [-2, 0, 2, 4],
        "Difference vs B0 (pp)",
    )
    body.append(
        line(
            project_effect(0),
            195,
            project_effect(0),
            effect_axis_y,
            stroke=MUTED,
            width=1.5,
            dash="7 5",
        )
    )
    selected_ci = [100.0 * value for value in curve["selected_medical_paired_bootstrap_95_ci"]]
    forest_row(
        body,
        project_effect,
        285,
        "Controller dev",
        100.0 * float(curve["selected_medical_delta"]),
        selected_ci[0],
        selected_ci[1],
        float(curve["selected_medical_mcnemar_exact_two_sided_p"]),
        PURPLE,
        label_x=810,
    )
    confirmation_ci = list(map(float, p10["paired_bootstrap_95_ci_percentage_points"]))
    forest_row(
        body,
        project_effect,
        410,
        "B2 confirmation",
        float(p10["delta_percentage_points"]),
        confirmation_ci[0],
        confirmation_ci[1],
        float(p10["mcnemar_exact_two_sided_p"]),
        ORANGE,
        label_x=810,
    )
    return document(
        "B2 development dose response and isolated confirmation",
        "Medical and General Controller deltas are plotted across B2 accepted checkpoints. A paired-effect forest plot contrasts the development-selected step with the six-hundred-question confirmation isolated from B2 training and model selection.",
        body,
        1200,
        600,
    )


def read_public_data(results_dir: Path, router_config_path: Path) -> dict:
    verify_result_hashes(results_dir)
    sft = load_json(results_dir / "sft_v3_confirmation.json")
    p10 = load_json(results_dir / "p10_confirmation.json")
    stage = load_json(results_dir / "stage120_controller.json")
    curve = load_json(results_dir / "b2_dose_curve.json")
    router_config = load_json(router_config_path)

    for payload in (sft, p10, stage, curve):
        if payload.get("final_access_count") != 0:
            raise ValueError(f"refusing to render after final access: {payload.get('artifact_kind')}")
    if sft["total"] != p10["total"] or sft["base"]["correct"] != p10["base"]["correct"]:
        raise ValueError("confirmation artifacts do not share the same frozen Base reference")
    if stage["medical_total"] != curve["medical_total"] or stage["general_total"] != curve["general_total"]:
        raise ValueError("Controller artifacts use inconsistent denominators")
    if curve["selected_development_step"] not in {point["step"] for point in curve["points"]}:
        raise ValueError("selected B2 development checkpoint is absent from the dose curve")

    router = router_config["router"]
    general_floor = 100.0 * (float(router["general_baseline"]) - float(router["delta"]))
    return {
        "sft": sft,
        "p10": p10,
        "stage": stage,
        "curve": curve,
        "general_floor": general_floor,
    }


def generated_figures(data: dict) -> dict[str, str]:
    return {
        "opd-training-loop.svg": render_opd_training_loop(),
        "ca-opd-routing.svg": render_ca_routing(),
        "experiment-overview.svg": render_experiment_overview(data),
        "b2-dose-confirmation.svg": render_b2_dose_confirmation(data),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--router-config", type=Path, default=DEFAULT_ROUTER_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless committed SVG files match regenerated content",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = read_public_data(args.results_dir, args.router_config)
    figures = generated_figures(data)

    if args.check:
        mismatches: list[str] = []
        for name, expected in figures.items():
            path_value = args.output_dir / name
            if not path_value.exists() or path_value.read_text(encoding="utf-8") != expected:
                mismatches.append(name)
        if mismatches:
            raise SystemExit("README figure mismatch: " + ", ".join(mismatches))
        print(f"README FIGURE CHECKS PASSED ({len(figures)} files)")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in figures.items():
        target = args.output_dir / name
        target.write_text(content, encoding="utf-8")
        print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
