"""Loss functions for TinyFold.

Provides unified loss functions for structure prediction including:
- MSE losses with Kabsch alignment
- Distance consistency losses
- Geometry auxiliary losses (bond lengths, angles, dihedrals)
- Contact-based losses
- lDDT metrics
"""

from .contact import (
    ContactLoss,
    compute_contact_mask,
    contact_loss_atoms,
    contact_loss_centroids,
)
from .geometry import (
    BOND_ANGLES,
    BOND_LENGTHS_ANGSTROM,
    GeometryLoss,
    bond_angle_loss,
    bond_length_loss,
    dihedral_angle,
    get_normalized_bond_lengths,
    o_chirality_loss,
    omega_loss,
    virtual_cb_loss,
)
from .lddt import (
    compute_ilddt,
    compute_interface_mask,
    compute_lddt,
    compute_lddt_metrics,
)
from .mse import (
    compute_c_rmsd,
    compute_distance_consistency_loss,
    compute_mse_loss,
    compute_relative_distance_loss,
    compute_rmse,
    kabsch_align,
)

__all__ = [
    # MSE
    "kabsch_align",
    "compute_mse_loss",
    "compute_rmse",
    "compute_c_rmsd",
    "compute_relative_distance_loss",
    "compute_distance_consistency_loss",
    # Geometry
    "GeometryLoss",
    "bond_length_loss",
    "bond_angle_loss",
    "omega_loss",
    "o_chirality_loss",
    "virtual_cb_loss",
    "dihedral_angle",
    "BOND_LENGTHS_ANGSTROM",
    "get_normalized_bond_lengths",
    "BOND_ANGLES",
    # Contact
    "ContactLoss",
    "compute_contact_mask",
    "contact_loss_centroids",
    "contact_loss_atoms",
    # lDDT
    "compute_lddt",
    "compute_ilddt",
    "compute_lddt_metrics",
    "compute_interface_mask",
]
