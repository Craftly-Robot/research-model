"""Render a lightweight dataset operations dashboard."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def render_dashboard(report: dict[str, Any]) -> str:
    crawl = report.get("crawl", {})
    source = report.get("source_registry", {})
    license_filter = report.get("license_filter_report") or {}
    benchmark_filter = report.get("benchmark_contamination_filter_report") or {}
    near_dedup = report.get("near_dedup_report") or {}
    contamination = report.get("contamination_report") or {}
    quality = report.get("quality_report") or {}
    training_quality = report.get("training_quality_report") or {}
    source_quality = report.get("source_quality_report") or {}
    source_reputation = report.get("source_reputation_report") or {}
    source_budget = report.get("source_budget_plan") or {}
    tasks = report.get("task_report") or {}
    review = report.get("review_report") or {}
    feedback = report.get("feedback_report") or {}
    uploaded = report.get("uploaded_objects") or []
    rows = [
        ("Status", report.get("status", "unknown")),
        ("Dataset", report.get("dataset_id", "")),
        ("Version", report.get("version_id", "")),
        ("Sources", source.get("source_count", 0)),
        ("Seed URLs", source.get("url_count", 0)),
        ("Fetched", crawl.get("fetched", 0)),
        ("Accepted", crawl.get("accepted", 0)),
        ("Rejected", crawl.get("rejected", 0)),
        ("Duplicates", crawl.get("duplicate", 0)),
        ("License Accepted", license_filter.get("accepted", 0)),
        ("License Rejected", license_filter.get("rejected", 0)),
        ("Benchmark-Filtered Rows", benchmark_filter.get("rejected", 0)),
        ("Near Duplicates Removed", near_dedup.get("near_duplicates", 0)),
        ("Contamination Hits", len(contamination.get("hits", []))),
        ("Avg Quality Score", quality.get("avg_quality_score", 0)),
        ("Avg Training Quality Score", training_quality.get("avg_quality_score", 0)),
        ("Training Rows", training_quality.get("rows", 0)),
        ("Sources Scored", len(source_quality.get("sources", []))),
        ("Reputation Sources", len(source_reputation.get("sources", []))),
        ("Budgeted Sources", len(source_budget.get("budgets", []))),
        ("Allocated Crawl Docs", source_budget.get("allocated_total_docs", 0)),
        ("Extracted Tasks", tasks.get("extracted", 0)),
        ("Approved Tasks", review.get("approved", 0)),
        ("Feedback Items", len(feedback.get("recommendations", []))),
        ("Uploaded Objects", len(uploaded)),
    ]
    table = "\n".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    warnings = source.get("warnings", [])
    warning_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings) or "<li>None</li>"
    reputation_rows = ""
    for item in source_reputation.get("sources", [])[:20]:
        reputation_rows += (
            "<tr>"
            f"<td>{html.escape(str(item.get('source', '')))}</td>"
            f"<td>{html.escape(str(item.get('reputation_score', '')))}</td>"
            f"<td>{html.escape(str(item.get('action', '')))}</td>"
            f"<td>{html.escape(', '.join(item.get('reasons', [])))}</td>"
            "</tr>"
        )
    if not reputation_rows:
        reputation_rows = "<tr><td colspan=\"4\">No reputation report yet</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Craftly Data Platform Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #17202a; }}
    table {{ border-collapse: collapse; width: min(900px, 100%); }}
    th, td {{ border: 1px solid #d6dbdf; padding: 10px 12px; text-align: left; }}
    th {{ width: 240px; background: #f4f6f7; }}
    code {{ background: #f4f6f7; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Craftly Data Platform Dashboard</h1>
  <table>{table}</table>
  <h2>Top Source Reputation</h2>
  <table>
    <tr><th>Source</th><th>Score</th><th>Action</th><th>Reasons</th></tr>
    {reputation_rows}
  </table>
  <h2>Registry Warnings</h2>
  <ul>{warning_html}</ul>
</body>
</html>
"""


def write_dashboard(report: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard(report), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Craftly data pipeline dashboard HTML.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    print(write_dashboard(report, args.output))


if __name__ == "__main__":
    main()

