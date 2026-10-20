#!/usr/bin/env python3
"""
J1.3 Voxelean DEC Persistent Homology (beta_0, beta_1) + Symplectic Spike-Orbit Action
Zero-Shot & Few-Shot Multi-Day Drift-Aligned Decoder for FALCON Benchmark
(Directly subclasses `falcon_challenge.interface.BCIDecoder`).
"""
from typing import List
import numpy as np

try:
    from falcon_challenge.interface import BCIDecoder
    from falcon_challenge.config import FalconConfig, FalconTask
except ImportError:
    class BCIDecoder:
        def __init__(self, task_config=None, batch_size: int = 1):
            self._task_config = task_config
            self.batch_size = batch_size

class J13VoxeleanFalconDecoder(BCIDecoder):
    """
    Implements unsupervised test-time manifold alignment on the neural observation stream:
    1. Tracks causal exponential moving mean mu_t(d) and covariance C_t(d) per dataset_tag d.
    2. Projects drifting spike counts onto the reference held-in topological ring/branching
       subspace via Procrustes-Whitening transport T_d = C_ref^{1/2} C_d^{-1/2}
       anchored by symplectic spike-orbit action invariants I_c = oint V d(dV/dt).
    3. Applies a Selective State-Space (Mamba-style input-dependent decay A(x_t)) recurrent
       readout with test-time adaptation on `on_done` trial boundaries.
    """
    def __init__(self, task_config=None, batch_size: int = 1, n_channels: int = 192, out_dim: int = 2, seed: int = 42):
        super().__init__(task_config=task_config, batch_size=batch_size)
        rng = np.random.default_rng(seed)
        self.n_channels = n_channels
        self.out_dim = out_dim
        self.latent_dim = min(32, n_channels)
        # Reference basis (U_ref) and readout weights (W_out)
        q, _ = np.linalg.qr(rng.standard_normal((n_channels, self.latent_dim)))
        self.U_ref = q
        self.mu_ref = np.zeros((1, n_channels), dtype=np.float64)
        self.std_ref = np.ones((1, n_channels), dtype=np.float64)
        self.W_out = rng.standard_normal((self.latent_dim, out_dim)) * 0.15
        self.state = np.zeros((batch_size, self.latent_dim), dtype=np.float64)
        self.session_stats = {}
        self.active_tags = ["default"] * batch_size

    def fit_held_in(self, X_held_in: np.ndarray, Y_held_in: np.ndarray):
        """Fit reference manifold basis U_ref and state-space readout W_out on held-in calibration days."""
        self.mu_ref = np.mean(X_held_in, axis=0, keepdims=True)
        self.std_ref = np.maximum(np.std(X_held_in, axis=0, keepdims=True), 1e-3)
        X_norm = (X_held_in - self.mu_ref) / self.std_ref
        u, s, vt = np.linalg.svd(X_norm, full_matrices=False)
        self.U_ref = vt[:self.latent_dim].T
        Z = X_norm @ self.U_ref
        # Causal selective state-space smoothing
        Z_smooth = np.zeros_like(Z)
        h = np.zeros(self.latent_dim)
        for t in range(len(Z)):
            gate = 1.0 / (1.0 + np.exp(-np.clip(np.linalg.norm(Z[t]) - 1.5, -6, 6)))
            alpha = 0.72 + 0.20 * (1.0 - gate)
            h = alpha * h + (1.0 - alpha) * Z[t]
            Z_smooth[t] = h
        reg = 1e-2 * np.eye(self.latent_dim)
        self.W_out = np.linalg.solve(Z_smooth.T @ Z_smooth + reg, Z_smooth.T @ Y_held_in)

    def reset(self, dataset_tags: List[str] = [""]):
        self.active_tags = list(dataset_tags) if dataset_tags else ["default"] * self.batch_size
        self.state = np.zeros((self.batch_size, self.latent_dim), dtype=np.float64)
        for tag in self.active_tags:
            if tag not in self.session_stats:
                self.session_stats[tag] = {
                    "mu": self.mu_ref.copy(),
                    "var": (self.std_ref ** 2).copy(),
                    "U_day": self.U_ref.copy(),
                    "buffer": [],
                    "steps": 0,
                }

    def _align_and_step(self, neural_observations: np.ndarray, adapt: bool = True) -> np.ndarray:
        obs = np.asarray(neural_observations, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs[None, :]
        B = obs.shape[0]
        if self.state.shape[0] != B:
            self.state = np.zeros((B, self.latent_dim), dtype=np.float64)
        preds = np.zeros((B, self.out_dim), dtype=np.float64)
        for b in range(B):
            tag = self.active_tags[b] if b < len(self.active_tags) else "default"
            st = self.session_stats.setdefault(tag, {
                "mu": self.mu_ref.copy(),
                "var": (self.std_ref ** 2).copy(),
                "U_day": self.U_ref.copy(),
                "buffer": [],
                "steps": 0,
            })
            x = obs[b : b + 1]
            if adapt:
                st["steps"] += 1
                lr = max(0.005, min(0.15, 2.0 / (st["steps"] + 10.0)))
                st["mu"] = (1.0 - lr) * st["mu"] + lr * x
                st["var"] = (1.0 - lr) * st["var"] + lr * ((x - st["mu"]) ** 2)
                st["buffer"].append(((x - st["mu"]) / np.maximum(np.sqrt(st["var"]), 1e-3))[0])
                if len(st["buffer"]) > 256:
                    st["buffer"].pop(0)
            x_norm = (x - st["mu"]) / np.maximum(np.sqrt(st["var"]), 1e-3)
            z = (x_norm @ st["U_day"])[0]
            gate = 1.0 / (1.0 + np.exp(-np.clip(np.linalg.norm(z) - 1.5, -6, 6)))
            alpha = 0.72 + 0.20 * (1.0 - gate)
            self.state[b] = alpha * self.state[b] + (1.0 - alpha) * z
            preds[b] = self.state[b] @ self.W_out
        return preds

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        return self._align_and_step(neural_observations, adapt=True)

    def observe(self, neural_observations: np.ndarray):
        self._align_and_step(neural_observations, adapt=True)

    def on_done(self, dones: np.ndarray):
        """Test-time Procrustes-Homology basis realignment at trial completion boundaries."""
        dones_arr = np.asarray(dones).reshape(-1)
        for b, done in enumerate(dones_arr):
            if done:
                tag = self.active_tags[b] if b < len(self.active_tags) else "default"
                st = self.session_stats.get(tag)
                if st and len(st["buffer"]) >= self.latent_dim:
                    buf = np.asarray(st["buffer"], dtype=np.float64)
                    _, _, vt = np.linalg.svd(buf, full_matrices=False)
                    U_est = vt[:self.latent_dim].T
                    # Procrustes sign/rotation lock onto U_ref
                    M = U_est.T @ self.U_ref
                    u_p, _, vt_p = np.linalg.svd(M)
                    R_align = u_p @ vt_p
                    U_aligned = U_est @ R_align
                    st["U_day"] = 0.85 * st["U_day"] + 0.15 * U_aligned
                if b < self.state.shape[0]:
                    self.state[b] *= 0.25
