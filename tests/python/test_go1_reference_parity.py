from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import gobot


OPTIONAL_DEPENDENCY_SKIP_CODE = 77
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _skip(reason: str) -> int:
    print(f"Go1 reference parity skipped: {reason}")
    return OPTIONAL_DEPENDENCY_SKIP_CODE


def main() -> int:
    availability = gobot.rl.MuJoCoWarpProvider.availability()
    if not availability.available:
        return _skip(availability.reason)

    import torch

    if not torch.cuda.is_available():
        return _skip("Torch cannot access CUDA")

    from examples.go1.tools.compare_go1_traces import compare_traces
    from examples.go1.tools.export_gobot_trace import export_trace
    from examples.go1.tools.go1_parity import read_trace

    fixture = REPO_ROOT / "tests/fixtures/go1/reference_trace_v1.json"
    with tempfile.TemporaryDirectory(prefix="gobot-go1-parity-") as directory:
        candidate_path = Path(directory) / "candidate.json"
        export_trace(output=candidate_path, device="cuda:0")
        errors = compare_traces(read_trace(fixture), read_trace(candidate_path))
    if errors:
        print("Go1 reference parity failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
