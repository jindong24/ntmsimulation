#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from lung_pde.cli import add_simulation_arguments, simulation_main


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the calibrated mucus-velocity lung PDE simulation."
    )
    return add_simulation_arguments(parser, default_final_time=9.0)


def main(argv=None):
    return simulation_main(
        build_parser(),
        mode="velocity",
        authoritative_source="test-v4.py",
        repository_root=_REPOSITORY_ROOT,
        script_name="scripts/simulate_velocity.py",
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
