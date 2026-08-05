import argparse
import csv
import json
import os
import statistics


STAGE_PAIRS = [
    ("receive_to_decode_ms", "frame_received", "decode_done"),
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
    ("bag_write_ms", "bag_write_start", "bag_write_done"),
    ("preview_ms", "video_write_done", "preview_done"),
    ("process_total_ms", "process_start", "visualize_done"),
    ("frame_total_ms", "frame_received", "preview_done"),
    ("steering_change_interval_ms", "previous_steering_change", "steering_change_done"),
]

METRIC_FIELDS = [
    "steering_deg",
    "previous_steering_deg",
    "steering_delta_deg",
    "steering_changed",
    "steering_change_interval_ms",
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
                "server_timestamp": row.get("server_timestamp"),
                "wall_time_ns": row.get("wall_time_ns"),
                "perf_to_wall_offset_ns": row.get("perf_to_wall_offset_ns"),
            }
            for name, start_key, end_key in STAGE_PAIRS:
                out[name] = duration_ms(marks, start_key, end_key)
            for key in METRIC_FIELDS:
                value = row.get("metrics", {}).get(key, "")
                if out.get(key, "") == "":
                    out[key] = value
            yield out


def write_summary(rows, path):
    fieldnames = ["metric", "count", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        summary_metrics = [metric for metric, _, _ in STAGE_PAIRS]
        for metric in summary_metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) != ""]
            if not values:
                continue
            writer.writerow({
                "metric": metric,
                "count": len(values),
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 95),
                "min_ms": min(values),
                "max_ms": max(values),
            })


def main():
    parser = argparse.ArgumentParser(description="Analyze raw timing marks from sep_com2.py")
    parser.add_argument("input", help="Path to timing_marks.jsonl")
    parser.add_argument("--per_frame_csv", default=None,
                        help="Output CSV with one row per frame")
    parser.add_argument("--summary_csv", default=None,
                        help="Output CSV with mean/median/p95/min/max per stage")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.input)) or "."
    per_frame_csv = args.per_frame_csv or os.path.join(base_dir, "timing_per_frame.csv")
    summary_csv = args.summary_csv or os.path.join(base_dir, "timing_summary.csv")

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

    write_summary(rows, summary_csv)
    print(f"Wrote {per_frame_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()




#ช่วยเพิ่มการคำนวณระยะเวลาในการเปลี่ยนมุมเลี้ยวของค่า cmd_vel ที่คำนวณได้ไหม และค่า time stamp ที่ถูกบันทึกจะนำมาคำนวณในไฟล์ timing_cal.py ด้วย
