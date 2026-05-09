from pathlib import Path


def parse_model_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def resolve_models(one_model: str | None, model_list: Path | None) -> list[str]:
    if one_model:
        return [one_model]
    if model_list is None:
        raise ValueError("--model-list is required when --model is not provided")
    models = parse_model_list(model_list)
    if not models:
        raise ValueError("model list is empty")
    return models
