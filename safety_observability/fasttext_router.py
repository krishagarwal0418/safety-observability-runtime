from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class FastTextRouter:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.model: Any | None = None
        if self.path and self.path.exists():
            import fasttext

            fasttext.FastText.eprint = lambda *a, **k: None  # type: ignore[attr-defined]
            self.model = fasttext.load_model(str(self.path))

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, text: str) -> dict[str, Any]:
        started = time.time()
        scores = {"attack": 0.0, "moderation": 0.0, "safe": 0.0}
        if self.model is not None:
            labels, probs = self.model.predict(" ".join(text.split()), k=-1)
            for label, prob in zip(labels, probs):
                key = label[len("__label__"):] if label.startswith("__label__") else label
                if key in scores:
                    scores[key] = float(prob)
        return {"scores": scores, "latency_ms": round((time.time() - started) * 1000, 3)}
