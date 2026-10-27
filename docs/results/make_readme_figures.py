#!/usr/bin/env python3
"""Save the three README PNGs from the shared figure builders."""
from distill_figures import FIGS, accuracy_fig, architecture_fig, speed_fig

for name, builder in [
    ("DISTILLATION_ARCHITECTURE_2026-10-27.png", architecture_fig),
    ("DISTILLATION_ACCURACY_2026-10-27.png", accuracy_fig),
    ("DISTILLATION_SPEED_2026-10-27.png", speed_fig),
]:
    fig = builder()
    fig.savefig(FIGS / name, bbox_inches="tight")
    fig.clf()
    print("wrote", FIGS / name)
