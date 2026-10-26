"""4G-HDNNP (fourth-generation HDNNP) student distilled from PolarMACE.

Architecture follows the RuNNer 2.0 4G-HDNNP formulation
(J. Chem. Phys. 165, 152502 (2026); arXiv:2607.17978): a local descriptor
network supplies per-atom electronegativity chi and hardness J, a periodic
QEq/PME solve dresses the atoms with equilibrated charges, and the energy is
local short-range plus 1/2 q^T E_coul q. Forces come from a single adjoint
solve (the same construction as RuNNer2 SI Eq. 34).

This is an independent PyTorch implementation, not the RuNNer Fortran code.
"""
