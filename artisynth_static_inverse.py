#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible entry point — 구현은 6_static_inverse.py."""
from importlib import import_module

_mod = import_module("6_static_inverse")

init = _mod.init
load_targets_csv = _mod.load_targets
solve_frame = _mod.solve_frame
run_all = _mod.run_static_inverse
muscle_names = _mod.muscle_names
shutdown = _mod.shutdown

if __name__ == "__main__":
    try:
        init()
        run_all()
    finally:
        shutdown()
