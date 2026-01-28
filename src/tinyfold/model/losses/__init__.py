"""Loss functions for TinyFold.

Provides unified loss functions for structure prediction including:
- MSE losses with Kabsch alignment
- Distance consistency losses
- Geometry auxiliary losses (bond lengths, angles, dihedrals)
- Contact-based losses
- lDDT metrics
"""

from .mse import (
    kabsch_align,
    compute_mse_loss,
    compute_rmse,
    compute_relative_distance_loss,
    compute_distance_consistency_loss,
)
from .geometry import (
    GeometryLoss,
    bond_length_loss,
    bond_angle_loss,
    omega_loss,
    o_chirality_loss,
    virtual_cb_loss,
    dihedral_angle,
    BOND_LENGTHS_ANGSTROM,
    get_normalized_bond_lengths,
    BOND_ANGLES,
)
from .contact import (
    ContactLoss,
    compute_contact_mask,
    contact_loss_centroids,
    contact_loss_atoms,
)
from .lddt import (
    compute_lddt,
    compute_ilddt,
    compute_lddt_metrics,
    compute_interface_mask,
)

__all__ = [
    # MSE
    "kabsch_align",
    "compute_mse_loss",
    "compute_rmse",
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
