from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def fig_to_section(title: str, fig) -> str:
    return f"<h2>{title}</h2>\n" + fig.to_html(full_html=False, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fMRI preprocessing QC HTML report.")
    parser.add_argument("--qc-csv", required=True, help="Path to preproc_qc_metrics.csv")
    parser.add_argument("--status-csv", default="", help="Optional path to preproc_status.csv")
    parser.add_argument("--out-html", required=True, help="Output HTML path")
    parser.add_argument("--fd-threshold", type=float, default=0.5, help="FD threshold in mm")
    args = parser.parse_args()

    qc_csv = Path(args.qc_csv)
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    if not qc_csv.exists():
        raise FileNotFoundError(f"QC CSV not found: {qc_csv}")

    qc = pd.read_csv(qc_csv)
    if qc.empty:
        raise ValueError("QC CSV is empty.")

    qc["fd_bad_percent"] = qc["fd_bad_ratio"] * 100.0
    qc = qc.sort_values("mean_fd", ascending=False)

    fig_mean = px.bar(
        qc,
        x="subject_id",
        y="mean_fd",
        title="Mean FD by Subject",
        labels={"subject_id": "Subject", "mean_fd": "Mean FD (mm)"},
    )
    fig_mean.add_hline(y=args.fd_threshold, line_dash="dash", line_color="red")

    fig_max = px.bar(
        qc,
        x="subject_id",
        y="max_fd",
        title="Max FD by Subject",
        labels={"subject_id": "Subject", "max_fd": "Max FD (mm)"},
    )
    fig_max.add_hline(y=args.fd_threshold, line_dash="dash", line_color="red")

    fig_hist = px.histogram(
        qc,
        x="mean_fd",
        nbins=min(20, max(5, len(qc))),
        title="Distribution of Mean FD",
        labels={"mean_fd": "Mean FD (mm)"},
    )
    fig_scatter = px.scatter(
        qc,
        x="mean_fd",
        y="fd_bad_percent",
        text="subject_id",
        title="Mean FD vs High-Motion Ratio",
        labels={"mean_fd": "Mean FD (mm)", "fd_bad_percent": "FD > threshold (%)"},
    )

    table_html = qc.to_html(index=False)

    status_html = ""
    if args.status_csv:
        status_csv = Path(args.status_csv)
        if status_csv.exists():
            status_df = pd.read_csv(status_csv)
            status_html = "<h2>Preprocessing Status</h2>\n" + status_df.to_html(index=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>fMRI QC Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #555; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f3f3f3; }}
  </style>
</head>
<body>
  <h1>fMRI Preprocessing QC Report</h1>
  <div class="meta">FD threshold: {args.fd_threshold:.3f} mm</div>
  {fig_to_section("Mean FD", fig_mean)}
  {fig_to_section("Max FD", fig_max)}
  {fig_to_section("Mean FD Distribution", fig_hist)}
  {fig_to_section("Motion Scatter", fig_scatter)}
  <h2>QC Table</h2>
  {table_html}
  {status_html}
</body>
</html>
"""

    out_html.write_text(html, encoding="utf-8")
    print(f"Saved report: {out_html}")


if __name__ == "__main__":
    main()
