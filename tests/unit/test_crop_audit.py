"""Tests for the static crop auditor's E1.a PASS/KILL logic (experiment plan).

The auditor's numeric reduction -- the fraction of crops that keep too few
inter-chain contacts and the single-chain fraction -- decides whether
InterfaceCrop is fit to train on before any run is spent. That reduction is pure
and is what these tests pin. The crop geometry itself is exercised by
test_crop_hardening.py against the real cropper.
"""

import importlib.util
from pathlib import Path

import torch

# audit_crops lives under scripts/, not the installed package -- load by path.
_AUDIT = Path(__file__).resolve().parents[2] / "scripts" / "data" / "audit_crops.py"
_spec = importlib.util.spec_from_file_location("audit_crops", _AUDIT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def _two_chain_sample(na: int, nb: int, gap: float) -> dict:
    """A trivial two-chain sample whose chains sit ``gap`` A apart on x."""
    n = na + nb
    chain = torch.zeros(n, dtype=torch.long)
    chain[na:] = 1
    coords = torch.zeros(n, 4, 3)
    # CA is atom index 1; place chain A near x=0, chain B near x=gap.
    coords[:na, 1, 0] = 0.0
    coords[na:, 1, 0] = gap
    return {"chain_ids": chain, "coords_res": coords, "n_res": n}


def test_inter_chain_contacts_counts_close_pairs():
    # Chains 1 A apart -> every A-B CA pair is a contact (< 8 A).
    s = _two_chain_sample(3, 2, gap=1.0)
    assert audit._inter_chain_contacts(s, cutoff=8.0) == 3 * 2
    # Chains 50 A apart -> no contacts.
    far = _two_chain_sample(3, 2, gap=50.0)
    assert audit._inter_chain_contacts(far, cutoff=8.0) == 0


def test_verdict_pass_when_within_thresholds():
    # 100 crops, all keep both chains, all keep >=5 contacts -> PASS.
    v = audit.audit_verdict([1.0] * 100, [10] * 100, min_contacts=5, kill_frac=0.20)
    assert v["verdict"] == "PASS"
    assert v["frac_single_chain"] == 0.0
    assert v["frac_low_contact"] == 0.0


def test_verdict_kill_on_single_chain_fraction():
    # 40/100 crops single-chain (> 20%) -> KILL, reason mentions single-chain.
    both = [0.0] * 40 + [1.0] * 60
    contacts = [10] * 100
    v = audit.audit_verdict(both, contacts, min_contacts=5, kill_frac=0.20)
    assert v["verdict"] == "KILL"
    assert v["frac_single_chain"] == 0.40
    assert "single-chain" in v["reason"]


def test_verdict_kill_on_contact_starvation():
    # 30/100 crops keep <5 inter-chain contacts (> 20%) -> KILL.
    both = [1.0] * 100
    contacts = [1] * 30 + [10] * 70
    v = audit.audit_verdict(both, contacts, min_contacts=5, kill_frac=0.20)
    assert v["verdict"] == "KILL"
    assert abs(v["frac_low_contact"] - 0.30) < 1e-9
    assert "contact" in v["reason"]


def test_verdict_boundary_is_strict_greater_than():
    # Exactly 20% starved is NOT > kill_frac -> PASS (boundary not a kill).
    both = [1.0] * 100
    contacts = [0] * 20 + [10] * 80
    v = audit.audit_verdict(both, contacts, min_contacts=5, kill_frac=0.20)
    assert v["frac_low_contact"] == 0.20
    assert v["verdict"] == "PASS"
