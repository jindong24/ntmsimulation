#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./setup.sh
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    printf '%s\n' 'Activate the FEniCS conda environment first.' >&2
    exit 1
fi
mkdir -p "$(dirname "$NTM_UFC_LIBRARY")"
case "$(uname -s)" in
    Darwin) flags=(-dynamiclib) ;;
    Linux) flags=(-shared -fPIC) ;;
    *) printf '%s\n' 'Supported platforms: Linux and macOS.' >&2; exit 1 ;;
esac
"${CXX:-c++}" -std=c++17 -O3 "${flags[@]}" -I"$CONDA_PREFIX/include" \
    reproduce/solver/src/lung_pde/ufc_batch.cpp -o "$NTM_UFC_LIBRARY"
printf '%s\n' "$NTM_UFC_LIBRARY"
