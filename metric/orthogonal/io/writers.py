from pathlib import Path
import json
import pandas as pd


def write_outputs(output_dir: Path, model: str, metrics: dict, raw_records: list[dict], summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / f"{model}_metrics.csv", index=False)

    raw_df = pd.DataFrame(raw_records)
    raw_df.to_csv(output_dir / f"{model}_raw.csv", index=False)

    with (output_dir / f"{model}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
