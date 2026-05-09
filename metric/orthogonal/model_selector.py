from pathlib import Path


def parse_model_list(model_list_path: Path) -> list[str]:
    models: list[str] = []
    for line in model_list_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        models.append(item)
    return models


def resolve_models(model: str | None, model_list_path: Path | None) -> list[str]:
    if model:
        return [model]
    if model_list_path is None:
        raise ValueError("model_list_path is required when model is not provided")
    models = parse_model_list(model_list_path)
    if not models:
        raise ValueError("model list is empty")
    return models
