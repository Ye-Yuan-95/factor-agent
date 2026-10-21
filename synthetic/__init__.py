"""Synthetic ground-truth DGPs and gate ablation experiments (packaged for the agent).

Modules keep flat imports (from stage_a_fpr_power import ...) working both in-repo and
installed: each inserts its own directory on sys.path at import time.
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
