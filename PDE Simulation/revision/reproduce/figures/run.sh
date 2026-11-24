#!/bin/bash
set -eu

here="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" -B "$here/run_figures.py" "$@"
