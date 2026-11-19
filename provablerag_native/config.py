from dataclasses import asdict, dataclass
from math import isfinite
import os
from pathlib import Path
import shutil


def prover9_path(workspace):
    return Path(os.environ.get("PROVER9") or shutil.which("prover9") or
                workspace / "outputs/vendor/LADR-2009-11A/bin/prover9").expanduser().resolve()


@dataclass(frozen=True)
class Config:
    max_candidates: int = 16
    max_planning_expansions: int | None = None
    max_formalization_repairs: int = 2
    top_k: int = 5
    max_search_calls: int = 8
    max_acquisition_operations: int = 20
    attempts_per_binding: int = 2
    max_binding_branches: int = 4
    planning_tokens: int = 12000
    execution_tokens: int = 16000
    max_total_tokens: int | None = None
    max_model_calls: int = 64
    solver_wall_seconds: float = 1.0
    solver_total_seconds: float = 10.0
    compact_citations: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in {"max_total_tokens", "max_planning_expansions"} and value is None:
                continue
            if name == "compact_citations":
                valid = type(value) is bool
            elif name.endswith("_seconds"):
                valid = type(value) in (float, int) and isfinite(value) and (
                    value >= 0 if name == "solver_total_seconds" else value > 0)
            else:
                minimum = 1 if name in {"top_k", "max_binding_branches", "attempts_per_binding"} else 0
                valid = type(value) is int and value >= minimum
            if not valid:
                raise ValueError(f"invalid setting {name}={value!r}")
