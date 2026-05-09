from pathlib import Path
import json
import pandas as pd


def write_pair(output_dir: Path, pair_name: str, rows: list[dict], summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{pair_name}.csv", index=False)
    with (output_dir / f"{pair_name}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
