# Chirality & Geometry Constraints for Stage 2

## Overview

Add auxiliary losses to penalize chemically impossible backbone configurations.
Start with bond lengths, progressively add more constraints.

---

## Implementation Phases

### Phase 1: Bond Lengths (START HERE)
**File**: `scripts/models/geometry_losses.py`

```python
BOND_LENGTHS = {
    'N_CA': 1.458,   # Within residue
    'CA_C': 1.525,   # Within residue
    'C_O': 1.229,    # Within residue
    'C_N': 1.329,    # Peptide bond (between residues)
}

def bond_length_loss(coords, mask):
    """
    coords: [B, L, 4, 3] - N=0, CA=1, C=2, O=3
    Returns: scalar loss
    """
    N, CA, C, O = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :], coords[..., 3, :]

    # Within-residue bonds
    d_N_CA = (N - CA).norm(dim=-1)   # [B, L]
    d_CA_C = (CA - C).norm(dim=-1)   # [B, L]
    d_C_O = (C - O).norm(dim=-1)     # [B, L]

    # Peptide bonds (C[i] to N[i+1])
    d_C_N = (C[:, :-1] - N[:, 1:]).norm(dim=-1)  # [B, L-1]

    # MSE from expected
    loss = (d_N_CA - 1.458)**2 + (d_CA_C - 1.525)**2 + (d_C_O - 1.229)**2
    loss_peptide = (d_C_N - 1.329)**2

    return loss.mean() + loss_peptide.mean()
```

---

### Phase 2: Bond Angles

```python
BOND_ANGLES = {
    'N_CA_C': 111.0,    # Tetrahedral at CA
    'CA_C_O': 121.0,    # sp2 carbonyl
    'CA_C_N': 117.0,    # sp2 peptide
    'C_N_CA': 121.0,    # sp2 peptide
}

def bond_angle(p0, p1, p2):
    """Angle at p1 in degrees."""
    v1 = F.normalize(p0 - p1, dim=-1)
    v2 = F.normalize(p2 - p1, dim=-1)
    cos_angle = (v1 * v2).sum(dim=-1).clamp(-1, 1)
    return torch.acos(cos_angle) * 180 / math.pi

def bond_angle_loss(coords, mask):
    N, CA, C, O = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :], coords[..., 3, :]
    N_next = N[:, 1:]
    CA_next = CA[:, 1:]

    # Within-residue angles
    angle_N_CA_C = bond_angle(N, CA, C)      # [B, L]
    angle_CA_C_O = bond_angle(CA, C, O)      # [B, L]

    # Cross-residue angles (peptide)
    angle_CA_C_N = bond_angle(CA[:, :-1], C[:, :-1], N_next)  # [B, L-1]
    angle_C_N_CA = bond_angle(C[:, :-1], N_next, CA_next)     # [B, L-1]

    loss = (angle_N_CA_C - 111)**2 + (angle_CA_C_O - 121)**2
    loss_pep = (angle_CA_C_N - 117)**2 + (angle_C_N_CA - 121)**2

    return loss.mean() + loss_pep.mean()
```

---

### Phase 3: Omega Dihedral (Peptide Planarity)

```python
def dihedral_angle(p0, p1, p2, p3):
    """Dihedral angle p0-p1-p2-p3 in radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    m1 = torch.cross(n1, F.normalize(b2, dim=-1), dim=-1)

    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)

    return torch.atan2(y, x)

def omega_loss(coords, mask):
    """
    Omega = CA(i) - C(i) - N(i+1) - CA(i+1)
    Should be ~180° (trans) or ~0° (cis)
    """
    CA, C = coords[..., 1, :], coords[..., 2, :]
    N_next, CA_next = coords[:, 1:, 0, :], coords[:, 1:, 1, :]

    omega = dihedral_angle(CA[:, :-1], C[:, :-1], N_next, CA_next)  # [B, L-1]

    # Loss: distance from ±π (trans) or 0 (cis)
    # Most bonds are trans, so penalize deviation from π
    trans_dev = (torch.abs(omega) - math.pi)**2
    cis_dev = omega**2

    # Take min to allow either trans or cis
    loss = torch.min(trans_dev, cis_dev)

    return loss.mean()
```

---

### Phase 4: O Chirality (Peptide Plane Handedness)

```python
def o_chirality_loss(coords, mask):
    """
    Ensure O is on correct side of peptide plane.
    Plane defined by CA, C, N_next.
    O should be trans to CA_next (opposite side).
    """
    CA, C, O = coords[..., 1, :], coords[..., 2, :], coords[..., 3, :]
    N_next, CA_next = coords[:, 1:, 0, :], coords[:, 1:, 1, :]

    # Plane normal: (C - CA) × (N_next - C)
    v1 = C[:, :-1] - CA[:, :-1]
    v2 = N_next - C[:, :-1]
    normal = torch.cross(v1, v2, dim=-1)  # [B, L-1, 3]

    # O position relative to plane
    o_vec = O[:, :-1] - C[:, :-1]
    o_side = (o_vec * normal).sum(dim=-1)  # [B, L-1]

    # CA_next position relative to plane
    ca_vec = CA_next - C[:, :-1]
    ca_side = (ca_vec * normal).sum(dim=-1)  # [B, L-1]

    # O and CA_next should be on OPPOSITE sides (trans)
    # If same sign, penalize
    loss = F.relu(o_side * ca_side)  # Positive when same side

    return loss.mean()
```

---

### Phase 5: Virtual CB Chirality (L-amino acid enforcement)

```python
def virtual_cb_loss(coords, mask, glycine_mask=None):
    """
    Compute virtual CB position and check chirality.
    CB should be at ~-34° improper dihedral for L-amino acids.

    Skip for glycine (no CB).
    """
    N, CA, C = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]

    # Virtual CB: perpendicular to N-CA-C plane, tetrahedral geometry
    v1 = F.normalize(N - CA, dim=-1)
    v2 = F.normalize(C - CA, dim=-1)

    # Bisector of N-CA-C
    bisector = F.normalize(v1 + v2, dim=-1)

    # Perpendicular to plane
    perp = torch.cross(v1, v2, dim=-1)
    perp = F.normalize(perp, dim=-1)

    # CB direction: rotate bisector away from plane
    # For L-amino acids, CB is on specific side
    cb_dir = -bisector * 0.5 + perp * 0.866  # ~60° from bisector
    CB_virtual = CA + cb_dir * 1.52  # CB-CA bond length

    # Improper dihedral: N - CA - C - CB
    chi = dihedral_angle(N, CA, C, CB_virtual)

    # Expected: ~-34° = -0.59 rad for L-amino acids
    expected = -0.59
    loss = (chi - expected)**2

    if glycine_mask is not None:
        loss = loss * (~glycine_mask).float()

    return loss.mean()
```

---

## Combined Loss Function

```python
class GeometryLoss(nn.Module):
    def __init__(self,
                 bond_length_weight=1.0,
                 bond_angle_weight=0.1,
                 omega_weight=0.1,
                 o_chirality_weight=0.1,
                 cb_chirality_weight=0.05):
        super().__init__()
        self.weights = {
            'bond_length': bond_length_weight,
            'bond_angle': bond_angle_weight,
            'omega': omega_weight,
            'o_chirality': o_chirality_weight,
            'cb_chirality': cb_chirality_weight,
        }

    def forward(self, coords, mask=None, glycine_mask=None):
        """
        coords: [B, L, 4, 3]
        Returns: dict of individual losses + total
        """
        losses = {}

        losses['bond_length'] = bond_length_loss(coords, mask)
        losses['bond_angle'] = bond_angle_loss(coords, mask)
        losses['omega'] = omega_loss(coords, mask)
        losses['o_chirality'] = o_chirality_loss(coords, mask)
        losses['cb_chirality'] = virtual_cb_loss(coords, mask, glycine_mask)

        total = sum(self.weights[k] * v for k, v in losses.items())
        losses['total'] = total

        return losses
```

---

## Integration with train_resfold.py

```python
from models.geometry_losses import GeometryLoss

# In main():
geom_loss_fn = GeometryLoss(
    bond_length_weight=args.bond_length_weight,
    bond_angle_weight=args.bond_angle_weight,
    omega_weight=args.omega_weight,
    o_chirality_weight=args.o_chirality_weight,
    cb_chirality_weight=args.cb_chirality_weight,
)

# In training loop (stage2_only or end_to_end):
if atoms_pred is not None:
    geom_losses = geom_loss_fn(atoms_pred, mask_res)
    loss = loss + args.geom_weight * geom_losses['total']
```

---

## Implementation Order

1. **Create `geometry_losses.py`** with `bond_length_loss()` only
2. **Test** that loss decreases during training
3. **Add** `bond_angle_loss()`, test
4. **Add** `omega_loss()`, test
5. **Add** `o_chirality_loss()`, test
6. **Add** `virtual_cb_loss()`, test (optional, may not be needed)

---

## Expected Impact

| Constraint | What it Fixes |
|------------|---------------|
| Bond lengths | Prevents stretched/compressed bonds |
| Bond angles | Ensures tetrahedral CA, planar peptide |
| Omega dihedral | Enforces peptide planarity (trans/cis) |
| O chirality | Prevents flipped carbonyl orientation |
| CB chirality | Enforces L-amino acid handedness |

---

## Notes

- All losses should be normalized by valid residue count
- Handle edge cases (first/last residue for peptide bonds)
- Glycine has no CB chirality constraint
- Bond lengths are in normalized coordinates (divide expected by std)
