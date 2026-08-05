import argparse
import csv
from datetime import datetime
import json
import os
import statistics


STAGE_PAIRS = [
    ("capture_ms", "capture_start", "capture_done"),
    ("preprocess_ms", "process_start", "preprocess_done"),
    ("inference_ms", "preprocess_done", "inference_done"),
    ("colorize_ms", "inference_done", "colorize_done"),
    ("mask_ms", "colorize_done", "mask_done"),
    ("nav_points_ms", "mask_done", "nav_points_done"),
    ("steering_calc_ms", "nav_points_done", "steering_calc_done"),
    ("cmd_vel_publish_ms", "cmd_vel_publish_start", "cmd_vel_publish_done"),
    ("navigation_ms", "nav_points_done", "navigation_done"),
    ("visualize_ms", "navigation_done", "visualize_done"),
    ("video_write_ms", "visualize_done", "video_write_done"),
    ("preview_ms", "video_write_done", "preview_done"),
    ("process_total_ms", "process_start", "visualize_done"),
    ("capture_to_cmd_vel_ms", "capture_start", "cmd_vel_publish_done"),
    ("loop_total_ms", "loop_start", "loop_done"),
    ("frame_total_ms", "capture_start", "preview_done"),
    ("steering_change_interval_ms", "previous_steering_change", "steering_change_done"),
]

METRIC_FIELDS = [
    "steering_deg",
    "previous_steering_deg",
    "steering_delta_deg",
    "steering_changed",
    "steering_change_interval_ms",
]

SUMMARY_FIELDNAMES = ["metric", "count", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"]

REPORT_STAGES = [
    ("Capture", "capture_ms"),
    ("Preprocess", "preprocess_ms"),
    ("Inference", "inference_ms"),
    ("Colorize", "colorize_ms"),
    ("Mask Clean", "mask_ms"),
    ("Navigation", "navigation_ms"),
    ("Visualize", "visualize_ms"),
    ("Video Write", "video_write_ms"),
    ("Preview", "preview_ms"),
]


def duration_ms(marks, start_key, end_key):
    start = marks.get(start_key)
    end = marks.get(end_key)
    if start is None or end is None:
        return ""
    return (end - start) / 1_000_000.0


def percentile(values, pct):
    if not values:
        return ""
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct / 100.0)
    return ordered[idx]


def fmt_ms(value):
    if value == "":
        return ""
    return f"{float(value):.3f}"


def fmt_num(value):
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.3f}"


def build_summary(rows):
    summary = []
    for metric, _, _ in STAGE_PAIRS:
        values = [float(row[metric]) for row in rows if row.get(metric) != ""]
        if not values:
            continue
        summary.append({
            "metric": metric,
            "count": len(values),
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 95),
            "min_ms": min(values),
            "max_ms": max(values),
        })
    return summary


def render_table(title, headers, rows):
    if not rows:
        return ""

    table_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in table_rows))
        for idx, header in enumerate(headers)
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [
        "",
        title,
        border,
        "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |",
        border,
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    lines.append(border)
    return "\n".join(lines)


def print_table(title, headers, rows):
    text = render_table(title, headers, rows)
    if text:
        print(text)


def summary_table_rows(summary):
    return [
        [
            row["metric"],
            row["count"],
            fmt_ms(row["mean_ms"]),
            fmt_ms(row["median_ms"]),
            fmt_ms(row["p95_ms"]),
            fmt_ms(row["min_ms"]),
            fmt_ms(row["max_ms"]),
        ]
        for row in summary
    ]


def steering_table_rows(rows):
    steering_values = [
        float(row["steering_deg"]) for row in rows
        if row.get("steering_deg") not in ("", None)
    ]
    steering_delta_values = [
        float(row["steering_delta_deg"]) for row in rows
        if row.get("steering_delta_deg") not in ("", None)
    ]
    changed_count = sum(1 for row in rows if row.get("steering_changed") is True)
    steering_interval_values = [
        float(row["steering_change_interval_ms"]) for row in rows
        if row.get("steering_change_interval_ms") not in ("", None)
    ]

    table_rows = []
    if steering_values:
        table_rows.append([
            "steering_deg",
            len(steering_values),
            fmt_num(statistics.fmean(steering_values)),
            fmt_num(statistics.median(steering_values)),
            fmt_num(min(steering_values)),
            fmt_num(max(steering_values)),
        ])
    if steering_delta_values:
        table_rows.append([
            "steering_delta_deg",
            len(steering_delta_values),
            fmt_num(statistics.fmean(steering_delta_values)),
            fmt_num(statistics.median(steering_delta_values)),
            fmt_num(min(steering_delta_values)),
            fmt_num(max(steering_delta_values)),
        ])
    if steering_interval_values:
        table_rows.append([
            "steering_change_interval_ms",
            len(steering_interval_values),
            fmt_ms(statistics.fmean(steering_interval_values)),
            fmt_ms(statistics.median(steering_interval_values)),
            fmt_ms(min(steering_interval_values)),
            fmt_ms(max(steering_interval_values)),
        ])
    if steering_values:
        table_rows.append([
            "steering_changed_frames",
            changed_count,
            f"{changed_count / len(rows) * 100.0:.1f}%",
            "",
            "",
            "",
        ])
    return table_rows


def wall_time_text(row):
    wall_time_ns = row.get("wall_time_ns")
    if wall_time_ns in ("", None):
        return ""
    return datetime.fromtimestamp(int(wall_time_ns) / 1_000_000_000.0).strftime("%Y-%m-%d %H:%M:%S")


def bar(value, total, width=20):
    if value == "" or total <= 0:
        return ""
    filled = round(float(value) / total * width)
    return "#" * min(width, max(0, filled))


def render_frame_block(row):
    total = row.get("loop_total_ms") or row.get("frame_total_ms") or row.get("process_total_ms") or ""
    total_value = float(total) if total != "" else 0.0
    latency = row.get("frame_total_ms") or row.get("loop_total_ms")
    latency_text = f"{float(latency):.0f}ms" if latency != "" else "-"
    fps = (1000.0 / total_value) if total_value > 0 else 0.0

    timestamp = wall_time_text(row)
    header_time = f" | {timestamp}" if timestamp else ""
    lines = [
        f"Frame #{row.get('frame', '-')}{header_time} | Latency: {latency_text}",
        "-" * 54,
        "  Step            Time(ms)       %",
        "  " + "-" * 36,
    ]

    for label, key in REPORT_STAGES:
        value = row.get(key)
        if value == "":
            continue
        value = float(value)
        pct = (value / total_value * 100.0) if total_value > 0 else 0.0
        lines.append(f"  {label:<13} {value:9.1f}ms {pct:6.1f}% {bar(value, total_value)}")

    lines.extend([
        "  " + "-" * 36,
        f"  TOTAL        {total_value:9.1f}ms  -> {fps:.1f} FPS",
    ])

    steering = row.get("steering_deg")
    if steering not in ("", None):
        delta = row.get("steering_delta_deg")
        interval = row.get("steering_change_interval_ms")
        delta_text = "" if delta in ("", None) else f" | delta {float(delta):.3f} deg"
        interval_text = "" if interval in ("", None) else f" | change interval {float(interval):.1f}ms"
        lines.append(f"  Steering     {float(steering):9.3f} deg{delta_text}{interval_text}")

    lines.extend([
        "-" * 54,
        "",
    ])
    return "\n".join(lines)


def sampled_rows(rows, every):
    if every <= 0:
        return []
    selected = [row for idx, row in enumerate(rows, start=1) if idx % every == 0]
    if rows and rows[-1] not in selected:
        selected.append(rows[-1])
    return selected


def build_report_text(rows, summary, frame_report_every):
    frame_values = [row.get("frame") for row in rows if row.get("frame") is not None]
    frame_start = min(frame_values) if frame_values else ""
    frame_end = max(frame_values) if frame_values else ""

    lines = [
        "Timing Report",
        "=" * 60,
        f"Input frames : {len(rows)}",
    ]
    if frame_start != "":
        lines.append(f"Frame range  : {frame_start} - {frame_end}")

    lines.append(render_table(
        "Stage Durations (ms)",
        ["stage", "count", "mean", "median", "p95", "min", "max"],
        summary_table_rows(summary),
    ))
    lines.append(render_table(
        "Steering Metrics",
        ["metric", "count", "mean", "median", "min", "max"],
        steering_table_rows(rows),
    ))

    frame_rows = sampled_rows(rows, frame_report_every)
    if frame_rows:
        lines.extend(["", "Per-frame Samples", "=" * 60])
        lines.extend(render_frame_block(row) for row in frame_rows)

    return "\n".join(line for line in lines if line is not None) + "\n"


def print_report(rows, summary):
    frame_values = [row.get("frame") for row in rows if row.get("frame") is not None]
    frame_start = min(frame_values) if frame_values else ""
    frame_end = max(frame_values) if frame_values else ""

    print()
    print("Timing Summary")
    print(f"Input frames : {len(rows)}")
    if frame_start != "":
        print(f"Frame range  : {frame_start} - {frame_end}")

    print_table(
        "Stage Durations (ms)",
        ["stage", "count", "mean", "median", "p95", "min", "max"],
        summary_table_rows(summary),
    )

    print_table(
        "Steering Metrics",
        ["metric", "count", "mean", "median", "min", "max"],
        steering_table_rows(rows),
    )


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            marks = row.get("perf_marks_ns", {})
            out = {
                "frame": row.get("frame"),
                "wall_time_ns": row.get("wall_time_ns"),
                "perf_to_wall_offset_ns": row.get("perf_to_wall_offset_ns"),
            }
            for name, start_key, end_key in STAGE_PAIRS:
                out[name] = duration_ms(marks, start_key, end_key)
            metrics = row.get("metrics", {})
            for key in METRIC_FIELDS:
                value = metrics.get(key, "")
                if out.get(key, "") == "":
                    out[key] = value
            yield out


def write_summary(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(build_summary(rows))


def write_text_report(rows, summary, path, frame_report_every):
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report_text(rows, summary, frame_report_every))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze raw timing marks from sep_com_combined_timing.py"
    )
    parser.add_argument("input", help="Path to combined_timing_marks.jsonl")
    parser.add_argument(
        "--per_frame_csv",
        default=None,
        help="Output CSV with one row per frame",
    )
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Output CSV with mean/median/p95/min/max per stage",
    )
    parser.add_argument(
        "--report_txt",
        default=None,
        help="Output readable text report",
    )
    parser.add_argument(
        "--frame_report_every",
        type=int,
        default=30,
        help="Write one per-frame detail block every N input rows. Use 0 to disable.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.input)) or "."
    per_frame_csv = args.per_frame_csv or os.path.join(base_dir, "timing_per_frame.csv")
    summary_csv = args.summary_csv or os.path.join(base_dir, "timing_summary.csv")
    report_txt = args.report_txt or os.path.join(base_dir, "timing_report.txt")

    rows = list(read_rows(args.input))
    if not rows:
        raise SystemExit(f"No timing rows found in {args.input}")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(per_frame_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows)
    write_summary(rows, summary_csv)
    write_text_report(rows, summary, report_txt, args.frame_report_every)
    print_report(rows, summary)
    print()
    print(f"Wrote {per_frame_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_txt}")


if __name__ == "__main__":
    main()
