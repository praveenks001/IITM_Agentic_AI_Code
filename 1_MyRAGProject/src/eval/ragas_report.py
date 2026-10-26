# ============================================================
# MyRAGProject - RAGAS Evaluation Report
# ============================================================

import csv
import json
from pathlib import Path


def save_ragas_report(rows, results, output_dir):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mode = next(iter(results.values())).mode

    csv_file = output_path / f"11_ragas_{mode}.csv"
    json_file = output_path / f"11_ragas_{mode}.json"

    report_rows = []

    for row in rows:

        question_id = row["id"]

        report_row = {
            "id": question_id,
            "question": row["question"],
            "answer": row["answer"],
            "ground_truth": row["ground_truth"],
            "contexts": json.dumps(
                row["contexts"],
                ensure_ascii=False
            ),
        }

        for metric_name, metric_result in results.items():
            report_row[metric_name] = (
                metric_result.per_row.get(question_id)
            )

        report_rows.append(report_row)

    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(report_rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(report_rows)

    summary = {
        "mode": mode,
        "total_questions": len(rows),
        "metrics": {
            name: {
                "aggregate": result.aggregate,
                "mode": result.mode
            }
            for name, result in results.items()
        }
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "questions": report_rows
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"RAGAS CSV saved: {csv_file}")
    print(f"RAGAS JSON saved: {json_file}")