from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import load_config, resolve_path


def download_models(config_path: str | None = None) -> dict[str, str]:
    cfg = load_config(config_path)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out: dict[str, str] = {}
    for name, spec in cfg["models"].items():
        repo_id = spec["repo_id"]
        local = resolve_path(spec["local_path"])
        local.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=local, token=token)
        out[name] = str(local)
        print(f"[download] {name}: {repo_id} -> {local}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(json.dumps(download_models(args.config), indent=2))


if __name__ == "__main__":
    main()
