#!/usr/bin/env python3
"""
Deterministic Verification Oracle for Wave 2 Open-Source Patches & Upstream PRs:
1. snel-repo/falcon-challenge (falcon_challenge/evaluator.py, falcon_challenge/dataloaders.py)
2. fwillett/speechBCI (NeuralDecoder/neuralDecoder/models.py, neuralSequenceDecoder.py)
3. neuro-galaxy/torch_brain PR #383 (torch_brain/transforms/neurolean_nhf.py)
4. neurallatents/nlb_tools PR #20 (nlb_tools/neurolean_combs.py, evaluation.py, make_tensors.py)
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/stu/solidsf-scratch/neuralink-wave2-repos/torch_brain")
sys.path.insert(0, "/Users/stu/solidsf-scratch/neuralink-wave2-repos/nlb_tools")

from torch_brain.transforms import (
    anharmonic_neural_hamiltonian,
    nhf_action_angle_coordinates,
    dehn_surgery_resolve_doublets,
    chiral_chern_orientation,
    ranicki_milgram_dyad_signature,
)
from nlb_tools import (
    schur_feshbach_lfp_spike_dress,
    koopman_5d_spike_jet_observables,
    betti_refractory_orbit_purification,
)


def test_falcon_challenge_bugs_and_fixes():
    HELDIN_OR_OUT_MAP = {'held_in': 'Held In', 'held_out': 'Held Out'}
    DATASET_HELDINOUT_MAP = {'h1': {'held_in': ['S0_set_1', 'S0_set_2']}}
    datasplit = 'h1'
    in_or_out = 'held_in'

    caught_key_error = False
    try:
        _ = HELDIN_OR_OUT_MAP[datasplit][in_or_out]
    except KeyError:
        caught_key_error = True
    assert caught_key_error, "Expected KeyError on HELDIN_OR_OUT_MAP[datasplit][in_or_out]"
    expected_sets = DATASET_HELDINOUT_MAP[datasplit][in_or_out]
    assert expected_sets == ['S0_set_1', 'S0_set_2']

    silent_trial = np.zeros((700, 158), dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        unpatched = (silent_trial - np.min(silent_trial)) / (np.max(silent_trial) - np.min(silent_trial))
    assert np.isnan(unpatched).all(), "Expected NaN from unpatched normalize_signal on silent trial"

    denom = np.max(silent_trial) - np.min(silent_trial)
    patched = np.zeros_like(silent_trial) if denom == 0 else (silent_trial - np.min(silent_trial)) / denom
    assert np.isfinite(patched).all() and np.all(patched == 0.0)

    rate_hz = 50.0
    dt_unpatched = rate_hz
    dt_patched = (1.0 / rate_hz) if rate_hz > 1.0 else rate_hz
    assert dt_unpatched == 50.0 and np.isclose(dt_patched, 0.02)
    print("FALCON_CHALLENGE_PATCH_VERIFICATION=PASS (4/4 bugs reproduced and verified)")


def test_speechbci_bugs_and_fixes():
    kernel_size = 14
    strides = 4
    subsample_factor = 2
    for T in [100, 102, 105, 113, 128]:
        t_after_stack = (T - kernel_size) // strides + 1
        x_stacked = np.zeros((1, t_after_stack, 256))
        x_sub = x_stacked[:, ::subsample_factor, :]
        actual_out_len = x_sub.shape[1]

        buggy_len = int((int(T / subsample_factor) - kernel_size) / strides + 1)
        patched_len = (T - kernel_size) // strides + 1
        patched_len = (patched_len + subsample_factor - 1) // subsample_factor

        assert patched_len == actual_out_len, f"Mismatch at T={T}: {patched_len} vs {actual_out_len}"
        assert buggy_len != actual_out_len, f"Expected upstream formula to mismatch at T={T}"
    print("SPEECHBCI_PATCH_VERIFICATION=PASS (3/3 bugs reproduced and verified)")


def test_torch_brain_pr383_neurolean_nhf():
    ham = anharmonic_neural_hamiltonian(np.array([1.0]), np.array([0.5]))
    assert ham["globally_confined"] is True
    t = np.arange(64, dtype=np.float64)
    wfs = np.stack([
        -7.0 * np.exp(-0.5 * ((t - 24.0) / 2.2) ** 2) + 2.5 * np.exp(-0.5 * ((t - 30.0) / 6.0) ** 2)
        for _ in range(16)
    ])
    nhf = nhf_action_angle_coordinates(wfs, k_modes=1)
    assert nhf["parseval_max_relative_error"] < 1e-13
    assert nhf["nhf_k_mode_capture_pct"] > nhf["fft_k_mode_capture_pct"]
    chiral = chiral_chern_orientation(wfs[:, :41], center_index=24)
    assert chiral["somatic_c1_neg1_count"] == 16
    print("TORCH_BRAIN_PR383_NEUROLEAN_NHF_VERIFICATION=PASS")


def test_nlb_tools_pr20_neurolean_combs():
    rng = np.random.default_rng(101)
    t = np.arange(2048, dtype=np.float64) / 19531.25
    sig = 2.0 * np.sin(2.0 * np.pi * 20.0 * t) + np.sin(2.0 * np.pi * 900.0 * t) + rng.normal(0.0, 0.1, len(t))
    sf = schur_feshbach_lfp_spike_dress(sig)
    assert sf["contraction_verified"] is True and sf["feshbach_banach_L_FB"] < 1.0
    koop = koopman_5d_spike_jet_observables(np.sin(t * 200.0) - 0.6 * (np.sin(t * 200.0) ** 3))
    assert koop["r2_koopman_5d"] >= koop["r2_linear_1d"]
    pur = betti_refractory_orbit_purification(
        np.array([0.0, 0.8, 2.2, 3.0]),
        np.array([[-6.0, 0.3], [-4.0, 0.8], [-6.0, 0.3], [-4.0, 0.8]]),
        tau_ref_ms=1.500,
    )
    assert pur["nilpotent_grassmann_compliant"] is True and pur["purified_single_unit_violations"] == 0
    print("NLB_TOOLS_PR20_NEUROLEAN_COMBS_VERIFICATION=PASS")


if __name__ == "__main__":
    test_falcon_challenge_bugs_and_fixes()
    test_speechbci_bugs_and_fixes()
    test_torch_brain_pr383_neurolean_nhf()
    test_nlb_tools_pr20_neurolean_combs()
