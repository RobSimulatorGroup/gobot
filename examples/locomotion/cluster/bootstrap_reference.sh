#!/usr/bin/env bash
set -euo pipefail
GOBOT_ROOT="${GOBOT_PROJECT_ROOT:-$PWD}"
GOBOT_REFERENCE_ENV="${GOBOT_REFERENCE_ENV:-$GOBOT_ROOT/examples/locomotion/logs/supermimic_reference/venv_317}"
cd "$GOBOT_ROOT"
if [[ -e "$GOBOT_REFERENCE_ENV/pyvenv.cfg" ]]; then
  printf "%s\n" "Reference environment already exists; choose a fresh GOBOT_REFERENCE_ENV." >&2
  exit 2
fi
uv venv --python 3.13 "$GOBOT_REFERENCE_ENV"
uv pip install --link-mode copy --python "$GOBOT_REFERENCE_ENV/bin/python" -r examples/locomotion/cluster/requirements-reference-317.txt
"$GOBOT_REFERENCE_ENV/bin/python" -m examples.locomotion.prepare_tracking_reference --patch-installed --output "$GOBOT_REFERENCE_ENV"
