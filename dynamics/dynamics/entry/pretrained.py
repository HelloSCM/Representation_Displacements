import sys
from pathlib import Path
import importlib.util


def run_pretrained(args: list[str]) -> None:
    root = Path(__file__).resolve().parents[2]
    train_file = root / "pretrained" / "train_distill.py"
    spec = importlib.util.spec_from_file_location("pretrained_train_distill", train_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {train_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = ["train_distill.py", *args]
    mod.main()
