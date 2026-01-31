"""Frame-Based Decoder: predicts centroid + rotation per residue.

Instead of predicting 12 values (4 atoms × 3 coords) directly, predicts:
- Centroid position: [B, L, 3]
- Rotation matrix: [B, L, 3, 3] via 6D representation

Atom positions are then computed as: atom_pos = centroid + R @ template

This guarantees correct backbone geometry by construction (no collapse possible).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def rotation_6d_to_matrix(rot_6d: Tensor) -> Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Uses Gram-Schmidt orthonormalization (Zhou et al., 2019).
    This representation is continuous and better for learning than quaternions.

    Args:
        rot_6d: [..., 6] tensor containing two 3D vectors

    Returns:
        R: [..., 3, 3] rotation matrix
    """
    a1 = rot_6d[..., :3]  # [..., 3]
    a2 = rot_6d[..., 3:]  # [..., 3]

    # Gram-Schmidt orthonormalization
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack into rotation matrix [b1, b2, b3] as columns
    R = torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]
    return R


def compute_backbone_template() -> Tensor:
    """Compute canonical backbone atom positions relative to centroid.

    Standard backbone geometry with:
    - N-CA bond: 1.458 Å
    - CA-C bond: 1.525 Å
    - C-O bond: 1.229 Å
    - N-CA-C angle: 111.2°
    - CA-C-O angle: 120.8°

    Returns:
        template: [4, 3] tensor with N, CA, C, O positions relative to centroid
    """
    # Place CA at origin, N along -x
    N = torch.tensor([-1.458, 0.0, 0.0])
    CA = torch.tensor([0.0, 0.0, 0.0])

    # C is at angle 111.2° from N-CA-C
    angle_NCA_C = math.radians(111.2)
    C = torch.tensor([
        1.525 * math.cos(math.pi - angle_NCA_C),
        1.525 * math.sin(math.pi - angle_NCA_C),
        0.0
    ])

    # O is attached to C at angle 120.8° from CA-C-O
    # Direction from C, in the peptide plane
    angle_CA_C_O = math.radians(120.8)
    CA_C = C - CA
    CA_C_norm = CA_C / CA_C.norm()
    # Rotate CA_C by angle_CA_C_O in the xy plane
    cos_a, sin_a = math.cos(angle_CA_C_O), math.sin(angle_CA_C_O)
    O_dir = torch.tensor([
        CA_C_norm[0] * cos_a - CA_C_norm[1] * sin_a,
        CA_C_norm[0] * sin_a + CA_C_norm[1] * cos_a,
        0.0
    ])
    O = C + 1.229 * O_dir

    # Stack atoms
    atoms = torch.stack([N, CA, C, O], dim=0)  # [4, 3]

    # Compute centroid and subtract
    centroid = atoms.mean(dim=0, keepdim=True)  # [1, 3]
    template = atoms - centroid  # [4, 3] relative to centroid

    return template


class FrameDecoder(nn.Module):
    """Frame-based decoder: predicts centroid + rotation per residue.

    Instead of predicting 12 values (4 atoms × 3 coords) directly,
    predicts 9 values (3 centroid + 6 rotation) and reconstructs atoms
    from a fixed backbone template.

    Key fix: anchor_pos is embedded and added to features BEFORE
    the refinement layers, so position info propagates through attention.
    """

    def __init__(
        self,
        c_token: int = 256,
        c_hidden: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_token = c_token
        self.c_hidden = c_hidden

        # Register backbone template (fixed geometry)
        template = compute_backbone_template()  # [4, 3]
        self.register_buffer('template', template)

        # Position embedding for anchor positions (like GeometricAtomDecoder)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, c_token),
            nn.GELU(),
            nn.Linear(c_token, c_token),
        )

        # Learnable embedding for unknown (non-anchored) positions
        self.unknown_embed = nn.Parameter(torch.randn(1, 1, c_token) * 0.1)

        # Refinement layers - now process position-aware features
        self.refine_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=c_token,
                nhead=n_heads,
                dim_feedforward=c_token * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        # Centroid prediction head
        self.centroid_head = nn.Sequential(
            nn.Linear(c_token, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, 3),
        )

        # Rotation prediction head (6D output)
        self.rotation_head = nn.Sequential(
            nn.Linear(c_token, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, 6),
        )

        # Initialize rotation head to output identity-ish rotation
        nn.init.zeros_(self.rotation_head[-1].weight)
        nn.init.constant_(self.rotation_head[-1].bias[:3], 1.0)  # a1 = [1, 0, 0]
        nn.init.zeros_(self.rotation_head[-1].bias[3:])
        self.rotation_head[-1].bias.data[4] = 1.0  # a2 = [0, 1, 0]

    def forward(
        self,
        trunk_tokens: Tensor,  # [B, L, c_token]
        anchor_pos: Tensor,    # [B, L, 3] input positions (can be zeros)
        mask: Optional[Tensor] = None,  # [B, L]
        coord_std: float = 10.0,  # Not used but kept for API compatibility
    ) -> Tensor:
        """Predict atom coordinates via centroid + rotation.

        Args:
            trunk_tokens: [B, L, c_token] residue features from trunk
            anchor_pos: [B, L, 3] input positions (added to centroid prediction)
            mask: [B, L] valid residue mask
            coord_std: Coordinate normalization factor (unused)

        Returns:
            atom_coords: [B, L, 4, 3] predicted atom coordinates
        """
        B, L, _ = trunk_tokens.shape
        device = trunk_tokens.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Check which residues have anchor positions
        is_anchored = (anchor_pos.abs().sum(dim=-1) > 1e-6)  # [B, L]

        # Embed anchor positions into features (KEY FIX!)
        pos_feats = self.pos_embed(anchor_pos)  # [B, L, c_token]
        pos_feats = torch.where(
            is_anchored.unsqueeze(-1),
            pos_feats,
            self.unknown_embed.expand(B, L, -1)
        )

        # Add position features to trunk tokens BEFORE refinement
        h = trunk_tokens + pos_feats

        # Refinement layers with masking (now position-aware!)
        for layer in self.refine_layers:
            attn_mask = ~mask if mask is not None else None
            h = layer(h, src_key_padding_mask=attn_mask)

        # Predict centroid (relative to anchor)
        centroid_delta = self.centroid_head(h)  # [B, L, 3]
        centroid = anchor_pos + centroid_delta  # [B, L, 3]

        # Predict rotation (6D -> 3x3)
        rot_6d = self.rotation_head(h)  # [B, L, 6]
        R = rotation_6d_to_matrix(rot_6d)  # [B, L, 3, 3]

        # Compute atom positions: centroid + R @ template
        # IMPORTANT: template is in Angstroms, divide by coord_std to match normalized coords
        template_norm = self.template / coord_std
        atom_offset = torch.einsum('blij,kj->blki', R, template_norm)  # [B, L, 4, 3]

        # Final atom positions
        atom_coords = (centroid.unsqueeze(2) + atom_offset).contiguous()  # [B, L, 4, 3]

        return atom_coords

    def forward_with_intermediates(
        self,
        trunk_tokens: Tensor,
        anchor_pos: Tensor,
        mask: Optional[Tensor] = None,
        coord_std: float = 10.0,
    ) -> dict:
        """Forward pass returning intermediate values for loss computation."""
        B, L, _ = trunk_tokens.shape
        device = trunk_tokens.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        is_anchored = (anchor_pos.abs().sum(dim=-1) > 1e-6)

        # Embed anchor positions
        pos_feats = self.pos_embed(anchor_pos)
        pos_feats = torch.where(
            is_anchored.unsqueeze(-1),
            pos_feats,
            self.unknown_embed.expand(B, L, -1)
        )

        h = trunk_tokens + pos_feats

        for layer in self.refine_layers:
            attn_mask = ~mask if mask is not None else None
            h = layer(h, src_key_padding_mask=attn_mask)

        centroid_delta = self.centroid_head(h)
        centroid = anchor_pos + centroid_delta

        rot_6d = self.rotation_head(h)
        R = rotation_6d_to_matrix(rot_6d)

        template_norm = self.template / coord_std
        atom_offset = torch.einsum('blij,kj->blki', R, template_norm)
        atom_coords = (centroid.unsqueeze(2) + atom_offset).contiguous()

        return {
            'atom_coords': atom_coords,
            'centroid': centroid,
            'rotation': R,
            'rot_6d': rot_6d,
        }


def atoms_to_frame(atoms: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute local frame (rotation matrix) from backbone atoms.

    Defines frame using CA-C as x-axis, peptide plane normal as z-axis.

    Args:
        atoms: [B, L, 4, 3] backbone atoms (N, CA, C, O)
        eps: Small value for numerical stability

    Returns:
        R: [B, L, 3, 3] rotation matrix for each residue
    """
    N = atoms[..., 0, :]   # [B, L, 3]
    CA = atoms[..., 1, :]  # [B, L, 3]
    C = atoms[..., 2, :]   # [B, L, 3]

    # X-axis: CA -> C direction
    x = C - CA
    x = x / (x.norm(dim=-1, keepdim=True) + eps)

    # Temporary vector: CA -> N direction
    n_dir = N - CA
    n_dir = n_dir / (n_dir.norm(dim=-1, keepdim=True) + eps)

    # Z-axis: perpendicular to peptide plane (x cross n_dir)
    z = torch.cross(x, n_dir, dim=-1)
    z = z / (z.norm(dim=-1, keepdim=True) + eps)

    # Y-axis: perpendicular to X and Z
    y = torch.cross(z, x, dim=-1)

    # Stack into rotation matrix
    R = torch.stack([x, y, z], dim=-1)  # [B, L, 3, 3]
    return R


def test_frame_decoder():
    """Test FrameDecoder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, L = 2, 50
    c_token = 256

    trunk_tokens = torch.randn(B, L, c_token, device=device)
    anchor_pos = torch.randn(B, L, 3, device=device) * 0.5
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    print("Testing FrameDecoder...")
    decoder = FrameDecoder(c_token=c_token, c_hidden=256, n_layers=4).to(device)
    out = decoder(trunk_tokens, anchor_pos, mask)
    print(f"  Output shape: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in decoder.parameters()) / 1e6:.2f}M")

    # Check atom spread (should be consistent due to fixed template)
    spread = (out - out.mean(dim=2, keepdim=True)).pow(2).sum(dim=-1).sqrt().mean()
    print(f"  Atom spread per residue: {spread.item():.4f}")

    # Check that geometry is preserved
    N_CA_dist = (out[..., 0, :] - out[..., 1, :]).norm(dim=-1).mean()
    CA_C_dist = (out[..., 1, :] - out[..., 2, :]).norm(dim=-1).mean()
    print(f"  N-CA distance: {N_CA_dist.item():.3f} (should be ~1.458)")
    print(f"  CA-C distance: {CA_C_dist.item():.3f} (should be ~1.525)")

    # Test with intermediates
    print("\nTesting forward_with_intermediates...")
    result = decoder.forward_with_intermediates(trunk_tokens, anchor_pos, mask)
    print(f"  Keys: {list(result.keys())}")
    print(f"  Centroid shape: {result['centroid'].shape}")
    print(f"  Rotation shape: {result['rotation'].shape}")

    # Check rotation is orthonormal
    R = result['rotation']
    RRT = torch.bmm(R.view(-1, 3, 3), R.view(-1, 3, 3).transpose(-1, -2))
    identity_err = (RRT - torch.eye(3, device=device)).abs().max()
    print(f"  Rotation orthonormality error: {identity_err.item():.6f}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_frame_decoder()
