from pathlib import Path

def abs_path(p: str) -> str:
    return str(Path(p).expanduser().resolve())
