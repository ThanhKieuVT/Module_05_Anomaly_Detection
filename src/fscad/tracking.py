"""Thin MLflow wrapper that enforces the run conventions of the project plan.

Every run is tagged with phase, dataset, class, shots, seed, method and git_commit. Runs from a
dirty working tree are refused when ``require_clean`` is set (always the case for ``final=true``).
If MLflow is not installed or tracking is disabled, every call is a no-op.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from fscad.utils import git_commit, log

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(
        self,
        enabled: bool,
        uri: str | None,
        experiment: str,
        require_clean: bool = False,
        final: bool = False,
    ):
        self.enabled = enabled and mlflow is not None
        if enabled and mlflow is None:
            log.warning("mlflow not installed: tracking disabled (pip install -e '.[mlops]')")
        self.final = final
        self.commit, dirty = git_commit()
        if (require_clean or final) and dirty:
            raise RuntimeError("refusing to run with uncommitted changes (tracking.require_clean / final)")
        if self.enabled:
            if uri:
                mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)

    @classmethod
    def from_cfg(cls, cfg) -> Tracker:
        t = cfg.tracking
        return cls(t.enabled, t.uri, t.experiment, t.require_clean, t.final)

    @contextmanager
    def run(self, name: str, tags: dict, params: dict | None = None, nested: bool = False):
        if not self.enabled:
            yield None
            return
        tags = {
            **{k: str(v) for k, v in tags.items()},
            "git_commit": self.commit,
            "final": str(self.final).lower(),
        }
        with mlflow.start_run(run_name=name, tags=tags, nested=nested) as r:
            if params:
                flat = _flatten(params)
                mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})
            yield r

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.enabled:
            clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
            mlflow.log_metrics(clean, step=step)

    def log_dict(self, d: dict, name: str) -> None:
        if self.enabled:
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / name
                p.write_text(json.dumps(d, indent=2, default=str))
                mlflow.log_artifact(str(p))

    def log_artifact(self, path: str | Path) -> None:
        if self.enabled:
            mlflow.log_artifact(str(path))
