"""Build liquid-like periodic benchmark boxes of rigid molecules."""
from __future__ import annotations

import math

import numpy as np
import torch
from ase.build import molecule as ase_molecule
from ase import Atoms


def build_water_box(n_molecules: int = 1728, density: float = 0.0334,
                    seed: int = 20261023, jitter: float = 0.15) -> Atoms:
    """Cubic H2O box at the given atom density (atoms/A^3)."""
    rng = np.random.default_rng(seed)
    water = ase_molecule("H2O")
    base = water.get_positions() - water.get_positions().mean(axis=0)
    n_atoms = 3 * n_molecules
    L = (n_atoms / density) ** (1.0 / 3.0)
    # Random liquid-like molecular centres with a minimum separation.  The
    # neighbour statistics are then Poisson-like (the grid placement below
    # would underestimate the angular environment).
    centers = []
    min_sep = 2.6
    while len(centers) < n_molecules:
        c = rng.uniform(0.0, L, size=3)
        if centers:
            cc = np.asarray(centers)
            d = cc - c
            d -= np.round(d / L) * L
            if np.any(np.linalg.norm(d, axis=1) < min_sep):
                continue
        centers.append(c)
    positions = []
    for i in range(n_molecules):
        center = centers[i]
        # random rotation (quaternion)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        pos = base @ R.T + center
        pos += rng.normal(scale=jitter, size=pos.shape)
        positions.append(pos)
    numbers = np.tile([8, 1, 1], n_molecules)
    atoms = Atoms(numbers=numbers, positions=np.vstack(positions),
                  cell=[L, L, L], pbc=True)
    return atoms


def write_xyz(atoms: Atoms, path: str):
    from ase.io import write
    write(path, atoms, format="extxyz")
