#!/usr/bin/env python3
"""Build the "artaastro" competition — predict the 6 WORLD-EVENT measures (GDELT global-daily conflict/
cooperation shares, unified into the monthly atlas by build_world_measures.py) from the sidereal sky.

A thin wrapper over the SHARED make_topic500_comp builder (AQ_COMP=artaastro), so artaastro uses the
identical constrained sinc-GD model (non-negative weights + frequencies, circular phase), the last-20%
FUTURE split and the phase-r2 sky-fingerprint board as topic-500 — the same PHP scorer path.

Prereq: python3 analysis/build_world_measures.py   (writes the 6 measure CSVs into analysis/data_monthly/).
  python3 analysis/make_artaastro_comp.py
"""
import os, runpy

os.environ["AQ_COMP"] = "artaastro"
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_topic500_comp.py"), run_name="__main__")
