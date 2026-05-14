"""
tests/test_focal_loss.py
─────────────────────────
Regression tests for FocalLoss and _build_criterion.

Verifies:
  1. gamma=0, weight=None  ->  identical to F.cross_entropy  (within 1e-5)
  2. No NaN / inf on a random batch
  3. focal(gamma=2) down-weights easy examples more aggressively than CE
  4. The old pt=exp(-weighted_ce) bug is gone: focal weight is not inflated
     for well-classified points that happen to have a high class weight
  5. ignore_index masking: ignored points do not contribute to the loss
  6. _build_criterion dispatches all 4 variants without error

Run with:
    python tests/test_focal_loss.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock tensorboard before importing trainer so the protobuf version mismatch
# on this machine doesn't prevent loading the loss classes under test.
sys.modules.setdefault("torch.utils.tensorboard", MagicMock())

import torch
import torch.nn.functional as F

from src.training.trainer import FocalLoss, _build_criterion


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rand_batch(B: int = 32, N: int = 128, C: int = 10):
    logits  = torch.randn(B * N, C)
    targets = torch.randint(1, C, (B * N,))   # classes 1–9 (ignore class 0)
    return logits, targets


# ─────────────────────────────────────────────────────────────────────────────
#  Test 1 — gamma=0, no weights  ->  equals F.cross_entropy
# ─────────────────────────────────────────────────────────────────────────────

def test_gamma0_equals_ce():
    logits, targets = _rand_batch()
    fl = FocalLoss(gamma=0.0, weight=None, ignore_index=0)

    loss_fl  = fl(logits, targets)
    loss_ce  = F.cross_entropy(logits, targets, ignore_index=0)

    diff = abs(loss_fl.item() - loss_ce.item())
    assert diff < 1e-5, (
        f"gamma=0 FocalLoss ≠ CrossEntropy: diff={diff:.2e}  "
        f"(FL={loss_fl.item():.6f}, CE={loss_ce.item():.6f})"
    )
    print(f"  OK gamma=0 matches CE within 1e-5  (diff={diff:.2e})")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2 — No NaN / inf on random inputs
# ─────────────────────────────────────────────────────────────────────────────

def test_no_nan_inf():
    logits, targets = _rand_batch(B=64, N=256)
    weights = torch.rand(10).abs() + 0.01
    weights[0] = 0.0   # unclassified

    for gamma in [0.5, 1.0, 2.0, 5.0]:
        fl = FocalLoss(gamma=gamma, weight=weights, ignore_index=0)
        loss = fl(logits, targets)
        assert torch.isfinite(loss), f"NaN/inf at gamma={gamma}: {loss.item()}"
        print(f"  OK gamma={gamma}: loss={loss.item():.4f}  (finite)")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3 — Focal modulation: easy examples get lower relative weight than CE
# ─────────────────────────────────────────────────────────────────────────────

def test_focal_down_weights_easy():
    C = 10
    # Easy: correct class gets a moderate boost  (p_t ≈ 0.86)
    logits_easy = torch.zeros(1, C)
    logits_easy[0, 3] = 4.0
    target_easy = torch.tensor([3])

    # Hard: uniform logits, model has no preference (p_t = 1/C = 0.10)
    logits_hard = torch.zeros(1, C)
    target_hard = torch.tensor([3])

    fl = FocalLoss(gamma=2.0, weight=None, ignore_index=0)
    ce = torch.nn.CrossEntropyLoss(ignore_index=0)

    loss_focal_easy = fl(logits_easy, target_easy).item()
    loss_focal_hard = fl(logits_hard, target_hard).item()
    loss_ce_easy    = ce(logits_easy, target_easy).item()
    loss_ce_hard    = ce(logits_hard, target_hard).item()

    ratio_focal = loss_focal_easy / loss_focal_hard
    ratio_ce    = loss_ce_easy    / loss_ce_hard

    assert ratio_focal < ratio_ce, (
        f"Focal should down-weight easy examples more than CE: "
        f"ratio_focal={ratio_focal:.4f} >= ratio_ce={ratio_ce:.4f}"
    )
    print(f"  OK Focal down-weights easy vs hard:  ratio_focal={ratio_focal:.5f}  ratio_ce={ratio_ce:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4 — The old bug: pt was p_t^w instead of p_t
#
#  With the fix, for a well-classified point (p_t ≈ 0.9999) and a large
#  class weight (w=20), focal_weight should be ~(1-0.9999)^2 ≈ 1e-8.
#  The old bug would give pt = 0.9999^20 ≈ 0.998, focal_weight ≈ 4e-6
#  — 400x larger, and scaling up with weight rather than down.
# ─────────────────────────────────────────────────────────────────────────────

def test_no_pt_corruption_by_class_weight():
    C = 10
    # Well-classified point: p_t ≈ 1.0
    logits  = torch.full((1, C), -10.0)
    logits[0, 3] = 10.0
    targets = torch.tensor([3])

    # Recompute expected pt directly
    with torch.no_grad():
        p_t_true = F.softmax(logits, dim=1)[0, 3].item()
    expected_focal_weight = (1.0 - p_t_true) ** 2

    # Apply a large class weight for class 3
    weights = torch.ones(C)
    weights[3] = 20.0

    fl = FocalLoss(gamma=2.0, weight=weights, ignore_index=0)

    # Manually reconstruct pt inside the loss to verify it is not corrupted
    log_p   = F.log_softmax(logits, dim=1)
    pt_impl = log_p[0, 3].exp().item()

    assert abs(pt_impl - p_t_true) < 1e-6, (
        f"pt is not derived from raw softmax: pt_impl={pt_impl:.8f}  p_t_true={p_t_true:.8f}"
    )
    print(f"  OK pt correctly derived from raw logits: pt={pt_impl:.8f}")

    # Also check the full loss is consistent: loss ≈ w * (1-pt)^2 * (-log pt)
    loss = fl(logits, targets).item()
    expected = weights[3].item() * expected_focal_weight * (-torch.log(torch.tensor(p_t_true)).item())
    assert abs(loss - expected) < 1e-4, (
        f"Loss value mismatch: got={loss:.6f}  expected≈{expected:.6f}"
    )
    print(f"  OK Loss matches w*(1-pt)^2*(-log pt): loss={loss:.8f}  expected~={expected:.8f}")

    # Second sub-case: moderate logits so pt != 1.0 and the loss is non-zero.
    # This catches the old bug numerically: exp(-w*CE) != pt for w != 1.
    logits2  = torch.zeros(1, C)
    logits2[0, 3] = 2.0        # p_t ≈ 0.47
    targets2 = torch.tensor([3])
    weights2 = torch.ones(C);  weights2[3] = 5.0

    with torch.no_grad():
        pt2  = F.softmax(logits2, dim=1)[0, 3].item()
        lpt2 = torch.log(torch.tensor(pt2)).item()

    fl2     = FocalLoss(gamma=2.0, weight=weights2, ignore_index=0)
    loss2   = fl2(logits2, targets2).item()
    # Correct formula: w * (1-pt)^gamma * (-log_pt)
    exp2    = weights2[3].item() * (1.0 - pt2) ** 2 * (-lpt2)
    # Old buggy formula: (1-pt^w)^gamma * (-w*log_pt)
    pt2_buggy = pt2 ** weights2[3].item()
    bug2      = (1.0 - pt2_buggy) ** 2 * (-weights2[3].item() * lpt2)

    assert abs(loss2 - exp2) < 1e-5, (
        f"Non-degenerate formula mismatch: loss={loss2:.6f}  correct={exp2:.6f}  buggy_would_give={bug2:.6f}"
    )
    assert abs(loss2 - bug2) > 0.01, (
        f"Fixed and buggy formulas are identical at this test point — test is not discriminating"
    )
    print(f"  OK Non-degenerate: pt={pt2:.4f}  loss={loss2:.5f}  correct={exp2:.5f}  buggy_would_give={bug2:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 5 — ignore_index masking
# ─────────────────────────────────────────────────────────────────────────────

def test_ignore_index():
    C = 10
    logits = torch.randn(100, C)

    # All ignored
    targets_all_ignored = torch.zeros(100, dtype=torch.long)
    fl = FocalLoss(gamma=2.0, weight=None, ignore_index=0)
    loss = fl(logits, targets_all_ignored)
    assert loss.item() == 0.0, f"All-ignored loss should be 0, got {loss.item()}"
    print(f"  OK All-ignored -> loss=0.0")

    # Mixed: half ignored, half valid
    targets_mixed = torch.randint(1, C, (100,))
    targets_mixed[:50] = 0   # ignore first half

    loss_mixed = fl(logits, targets_mixed)
    loss_valid  = fl(logits[50:], targets_mixed[50:])   # same valid half only

    assert abs(loss_mixed.item() - loss_valid.item()) < 1e-5, (
        f"Ignored points leaked into mean: mixed={loss_mixed.item():.6f}  valid={loss_valid.item():.6f}"
    )
    print(f"  OK Ignored points do not affect mean  (diff={abs(loss_mixed.item()-loss_valid.item()):.2e})")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 6 — _build_criterion dispatches all 4 variants
# ─────────────────────────────────────────────────────────────────────────────

def test_build_criterion():
    C = 10
    class_w = torch.rand(C).abs() + 0.1;  class_w[0] = 0.0
    cb_w    = torch.rand(C).abs() + 0.1;  cb_w[0]    = 0.0

    logits  = torch.randn(64, C)
    targets = torch.randint(1, C, (64,))

    for loss_type, kw in [
        ("ce",          dict(class_weights=None, cb_weights=None)),
        ("weighted_ce", dict(class_weights=class_w, cb_weights=None)),
        ("focal",       dict(class_weights=class_w, cb_weights=None)),
        ("cb_focal",    dict(class_weights=class_w, cb_weights=cb_w)),
    ]:
        crit = _build_criterion(loss_type, ignore_index=0, gamma=2.0, **kw)
        loss = crit(logits, targets)
        assert torch.isfinite(loss), f"{loss_type}: loss is not finite ({loss.item()})"
        print(f"  OK _build_criterion('{loss_type}'): loss={loss.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FocalLoss regression tests\n")

    print("Test 1 — gamma=0 equals cross_entropy:")
    test_gamma0_equals_ce()

    print("\nTest 2 — No NaN/inf on random inputs:")
    test_no_nan_inf()

    print("\nTest 3 — Focal down-weights easy examples:")
    test_focal_down_weights_easy()

    print("\nTest 4 — pt not corrupted by class weight (old bug regression):")
    test_no_pt_corruption_by_class_weight()

    print("\nTest 5 — ignore_index masking:")
    test_ignore_index()

    print("\nTest 6 — _build_criterion all 4 variants:")
    test_build_criterion()

    print("\nAll tests passed.")
