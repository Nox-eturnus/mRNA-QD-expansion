"""mRNA_SQD: experimental-data benchmark, energy-model fixes, and SQD/SKQD
for QUBO-based RNA secondary-structure prediction.

The frozen baseline lives in ``../WISER-Moderna``; ``baseline_path()`` puts
it on ``sys.path`` so its modules (``genus``, ``qubo``, ...) can be reused
without copying.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(_ROOT, 'WISER-Moderna')
DATA_DIR = os.path.join(_ROOT, 'data')
RAW_DIR = os.path.join(_ROOT, 'raw')


def baseline_path() -> str:
    """Make the baseline modules importable and return their directory."""
    if BASELINE_DIR not in sys.path:
        sys.path.insert(0, BASELINE_DIR)
    return BASELINE_DIR
