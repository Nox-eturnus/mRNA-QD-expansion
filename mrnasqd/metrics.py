"""Structure metrics, identical in definition to the baseline's
``evaluate.compute_structure_metrics`` (TN over all C(N,2) position pairs),
without importing the baseline's quantum stack."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple


def structure_metrics(predicted: Iterable[Tuple[int, int]],
                      truth: Iterable[Tuple[int, int]],
                      seq_len: int) -> Dict[str, float]:
    p = {(min(i, j), max(i, j)) for i, j in predicted}
    t = {(min(i, j), max(i, j)) for i, j in truth}
    tp, fp, fn = len(p & t), len(p - t), len(t - p)
    tn = seq_len * (seq_len - 1) // 2 - tp - fp - fn
    sens = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(d) if d > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'sensitivity': sens,
            'ppv': ppv, 'f1': f1, 'mcc': mcc}
