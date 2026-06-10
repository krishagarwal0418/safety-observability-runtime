from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .config import load_config, resolve_path


def download_models(config_path: str | None = None) -> dict[str, str]:
    cfg = load_config(config_path)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out: dict[str, str] = {}
    for name, spec in cfg["models"].items():
        local = resolve_path(spec["local_path"])
        repo_id = spec.get("repo_id")
        if not repo_id:
            print(f"[download] skip {name}: no repo_id configured")
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        if name == "fasttext_router":
            filename = spec.get("filename", "router_head.ftz")
            src = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local)
        else:
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
