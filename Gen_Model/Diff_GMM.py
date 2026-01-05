"""
Differentiable Gaussian‑Mixture Data Transformer
================================================

This module provides a **Gaussian‑Mixture–based data transformer** that

* flattens multi‑modal continuous features into an almost standard‑normal
  space;
* optionally exposes a (soft or Gumbel‑Softmax) component indicator so
  downstream models may exploit mode information;
* is **fully differentiable in both directions** (forward and inverse);
* can be frozen after an unsupervised pre‑fit so its statistics remain
  fixed while gradients still flow through the mapping.

The design follows the recipe discussed in our conversation:
    x  →  [γ₁…γᴷ,  z]      (forward)
    [γ₁…γᴷ, z]  →  x̂       (inverse)
where γ collects posterior responsibilities and z is a standardised
residual.

Two public classes are provided
--------------------------------
* **`DiffGMM1D`**      — the exact scalar transformer (single feature).
* **`GMMDataTransformer`** — a thin wrapper that vectorises the scalar
  transformer over **D independent features**.  Per–feature Gaussians are
  factorised; the whole transformer is a single `nn.Module` so you can
  `.to(device)` once.

Example
-------
>>> tfm = GMMDataTransformer(n_components=5, n_features=3)
>>> tfm.fit(torch.randn(20_000, 3))      # unsupervised fit (freezes)
>>> x       = torch.randn(128, 3, requires_grad=True)
>>> rep     = tfm(x)
>>> x_hat   = tfm.inverse(rep)
>>> loss    = ((x_hat - x)**2).mean()
>>> loss.backward()                      # gradients into x, none into tfm

"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.mixture import GaussianMixture
except ImportError:  # keep the module import‑safe on minimal envs
    GaussianMixture = None  # type: ignore

LOG2PI = torch.log(torch.tensor(2.0 * torch.pi))


# ---------------------------------------------------------------------------
#                             1‑D  Transformer
# ---------------------------------------------------------------------------
class DiffGMM1D(nn.Module):
    """Differentiable scalar transformer with K‑component GMM.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components (K).
    init_mu, init_log_sigma, init_logits : torch.Tensor, shape (K,)
        Initial mixture parameters.  Provide them from an external GMM
        or random init; *they will be frozen* unless `trainable=True`.
    gumbel_softmax : bool, default ``False``
        If ``True`` output responsibilities via Gumbel‑Softmax + straight‑
        through estimator; if ``False`` output the soft posterior weights.
    tau : float, default ``1.0``
        Temperature for the Gumbel‑Softmax.
    trainable : bool, default ``False``
        Whether the mixture parameters may be updated by the optimiser.
    """

    def __init__(
        self,
        n_components: int,
        init_mu: torch.Tensor,
        init_log_sigma: torch.Tensor,
        init_logits: torch.Tensor,
        *,
        gumbel_softmax: bool = True,
        tau: float = 1.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        K = n_components
        # register – optionally as parameters
        factory = nn.Parameter if trainable else lambda t: nn.Parameter(t, requires_grad=False)
        self.mu      = factory(init_mu.clone().detach().reshape(K))
        self.log_s   = factory(init_log_sigma.clone().detach().reshape(K))
        self.logits  = factory(init_logits.clone().detach().reshape(K))

        self.K = K
        self.use_gs = gumbel_softmax
        self.register_buffer("_tau", torch.tensor(float(tau)))

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    @property
    def sigma(self) -> torch.Tensor:  # σ = exp(logσ)
        return self.log_s.exp()

    def _posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute component responsibilities γ(x) and per‑component z‑scores.

        Returns
        -------
        gamma : torch.Tensor, shape (B, K)
        z_k   : torch.Tensor, shape (B, K)
        """
        x = x.unsqueeze(-1)  # (B, 1)
        # log N(x | μ, σ²)
        log_prob = -0.5 * ((x - self.mu) / self.sigma) ** 2 - self.log_s - 0.5 * LOG2PI  # (B, K)
        log_mix  = F.log_softmax(self.logits, dim=-1)                                     # (K,)
        log_joint = log_prob + log_mix                                                    # (B, K)
        log_gamma = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)          # (B, K)
        gamma = log_gamma.exp()

        z_k = (x - self.mu) / self.sigma  # (B, K)
        return gamma, z_k

    # ------------------------------------------------------------------
    # forward / inverse
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward transform x → representation.

        Parameters
        ----------
        x : torch.Tensor, shape (B,)

        Returns
        -------
        rep : torch.Tensor, shape (B, K+1)
            Concatenation of responsibilities (soft or GS one‑hot) and a
            single scalar z = Σ γ_k z_k.
        """
        gamma, z_k = self._posterior(x)

        if self.use_gs:
            # —— numerically stable Gumbel‑Softmax ————————————————
            eps = 1e-8
            gamma_clamped = gamma.clamp_min(eps)              # avoid log(0) → −inf
            g = -torch.empty_like(gamma).exponential_().log() # Gumbel noise
            logits_gs = (gamma_clamped.log() - g) / self._tau
            y_soft = F.softmax(logits_gs, dim=-1)             # differentiable
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(-1, y_soft.argmax(-1, keepdim=True), 1.0)
            gamma_used = (y_hard - y_soft).detach() + y_soft  # straight‑through
        else:
            gamma_used = gamma

        z_scalar = (gamma_used * z_k).sum(-1, keepdim=True)  # (B,1)
        return torch.cat([gamma_used, z_scalar], dim=-1)     # (B, K+1)
    ''' 
    def inverse(self, rep: torch.Tensor) -> torch.Tensor:
        """Inverse mapping representation → x̂ (reconstruction)."""
        gamma_like, z = rep[..., : self.K], rep[..., -1:]
        x_hat = (gamma_like * (z * self.sigma + self.mu)).sum(-1)
        return x_hat
    '''
    def inverse(self, rep: torch.Tensor) -> torch.Tensor:
        """Inverse mapping (gamma_raw-logits → x̂)."""
        gamma_raw, z = rep[..., :self.K], rep[..., -1:]          # split
        gamma = torch.softmax(gamma_raw, dim=-1)                     # logits → probs

        # broadcast: (B,1)·(K,) → (B,K)  then weighted sum → (B,)
        x_hat = (gamma * (z * self.sigma + self.mu)).sum(-1)
        return x_hat

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------
    def freeze(self, also_buffers: bool = False) -> None:
        """Prevent all mixture parameters from receiving gradients."""
        for p in self.parameters():
            p.requires_grad_(False)
        if also_buffers:
            self._tau.requires_grad_(False)  # usually not trainable anyway


# ---------------------------------------------------------------------------
#                         Vector‑of‑Features Transformer
# ---------------------------------------------------------------------------
class GMMDataTransformer(nn.Module):
    """Apply *independent* DiffGMM1D transformers to D features.

    Attributes
    ----------
    scalers : nn.ModuleList[DiffGMM1D]
        One transformer per feature.
    K : int
        Number of components.
    D : int
        Number of features.
    flat_out : bool
        If ``True`` (default) the forward output is flattened to
        `(B, D*(K+1))`; if ``False`` the shape is `(B, D, K+1)`.
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        *,
        gumbel_softmax: bool = False,
        tau: float = 1.0,
        trainable: bool = False,
        flat_out: bool = True,
    ) -> None:
        super().__init__()
        self.K = n_components
        self.D = n_features
        self.flat_out = flat_out
        self._cfg = dict(gumbel_softmax=gumbel_softmax, tau=tau, trainable=trainable)

        # placeholder; user must call .fit() or .load_state_dict()
        self.scalers = nn.ModuleList([
            DiffGMM1D(self.K, torch.zeros(self.K), torch.zeros(self.K), torch.zeros(self.K), **self._cfg)
            for _ in range(self.D)
        ])
        self._fitted = False

    # ------------------------------------------------------------------
    # Unsupervised fitting (prototypically via sklearn)
    # ------------------------------------------------------------------
    def fit(self, data: torch.Tensor, *, device: Optional[torch.device] = None) -> "GMMDataTransformer":
        """Learn mixture parameters from raw *tensor* data.

        Parameters
        ----------
        data : torch.Tensor, shape (N, D)
        device : torch.device, optional
            If provided, move the resulting transformer to that device.
        """
        if data.ndim != 2 or data.shape[1] != self.D:
            raise ValueError(f"Expected data of shape (N,{self.D}); got {tuple(data.shape)}")
        if GaussianMixture is None:
            raise ImportError("scikit‑learn is required for .fit(); install 'scikit‑learn'.")

        scalers: Sequence[DiffGMM1D] = []
        np_data = data.detach().cpu().numpy()
        for d in range(self.D):
            gmm = GaussianMixture(self.K).fit(np_data[:, [d]])
            #print(f"[fit] Fitting GMM for feature {d + 1:>2}/{self.D}...")
            # extract parameters
            #print(f"[fit] GMM means: {gmm.means_.ravel()}")
            #print(f"[fit] GMM covariances: {gmm.covariances_.ravel() ** 0.5}")
            #print(f"[fit] GMM weights: {gmm.weights_}")
            mu0     = torch.tensor(gmm.means_.ravel(),        dtype=torch.float32)
            log_s0  = torch.log(torch.tensor(gmm.covariances_.ravel() ** 0.5, dtype=torch.float32))
            logits0 = torch.log(torch.tensor(gmm.weights_,     dtype=torch.float32))
            scalers.append(
                DiffGMM1D(self.K, mu0, log_s0, logits0, **self._cfg)
            )


        # replace placeholder list
        self.scalers = nn.ModuleList(scalers)
        self._fitted = True

        if device is not None:
            self.to(device)
        return self

    # ------------------------------------------------------------------
    # forward / inverse
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Transform *B×D* raw tensor to representation.

        Returns `(B, D*(K+1))` if `flat_out=True`, else `(B, D, K+1)`.
        """
        if not self._fitted:
            raise RuntimeError("Transformer must be .fit()‑ted (or loaded) before use.")
        if x.ndim != 2 or x.shape[1] != self.D:
            raise ValueError(f"Expected shape (B,{self.D}); got {tuple(x.shape)}")

        reps = [scaler(x[:, d]) for d, scaler in enumerate(self.scalers)]  # list of (B, K+1)
        rep_full = torch.stack(reps, dim=1)  # (B, D, K+1)
        if self.flat_out:
            rep_full = rep_full.flatten(start_dim=1)  # (B, D*(K+1))
        return rep_full

    def inverse(self, rep: torch.Tensor) -> torch.Tensor:
        """Inverse transform representation → raw.

        Accepts both flat and non‑flat shapes.
        """
        if not self._fitted:
            raise RuntimeError("Transformer must be .fit()‑ted (or loaded) before use.")

        if rep.ndim == 2 and rep.shape[1] == self.D * (self.K + 1):
            rep = rep.view(rep.shape[0], self.D, self.K + 1)  # un‑flatten
        elif rep.ndim != 3 or rep.shape[1:] != (self.D, self.K + 1):
            raise ValueError("Representation has incompatible shape")

        xs = [self.scalers[d].inverse(rep[:, d]) for d in range(self.D)]  # list of (B,)
        x_recon = torch.stack(xs, dim=1)  # (B, D)
        return x_recon

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """Freeze *all* mixture statistics."""
        for scaler in self.scalers:
            scaler.freeze()

    def unfreeze(self) -> None:
        """Make mixture parameters trainable again (rarely needed)."""
        for scaler in self.scalers:
            for p in scaler.parameters():
                p.requires_grad_(True)

    # convenient alias
    encode = forward
    decode = inverse


if __name__ == "__main__":
    import pandas as pd, torch
    df = pd.read_csv("../train-v2.csv").iloc[:,:-1]
    
    cat_names  = ["spg"] + [f"wp{i}" for i in range(8)]          #  9 categorical
    cont_names = [c for c in df.columns if c not in cat_names]   # 48 continuous here

    x_cont_df = df[cont_names]         # (N, 48) floats

    x_cont = torch.tensor(x_cont_df.values, dtype=torch.float32)
    cont_tfm = GMMDataTransformer(
        n_components   = 10,            # K for every feature
        n_features     = x_cont.shape[1],
        gumbel_softmax = True,
        tau            = 1.0,
    ).fit(x_cont)
    cont_tfm.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Move the frozen transformers onto the execution device
    cont_tfm = cont_tfm.to(device)

    # B = batch size
    batch  = df.sample(128)
    x_cont_df = batch[cont_names]  # (N, 48) floats

    print("Batch shape :", batch.shape)
    x_cont = torch.tensor(x_cont_df.values, dtype=torch.float32, device=device)
    x_cont = x_cont.clone().detach().to(device).requires_grad_(True)

    rep_cont_part = cont_tfm(x_cont)      # has grad_fn

    sigma_noise = 1e-2                    # try 1e-2 … 1e-1
    noise = sigma_noise * torch.randn_like(rep_cont_part)
    rep_noisy = rep_cont_part + noise     # still requires_grad=True

    x_cont_hat = cont_tfm.inverse(rep_noisy)
    loss       = ((x_cont_hat - x_cont)**2).mean()


    # ask autograd to keep .grad for a non-leaf
    rep_cont_part.retain_grad()

    loss.backward()
    print("Loss: ", loss.item())

    print("rep_cont_part.grad is None? ", rep_cont_part.grad)  # False
    print("x_cont.grad norm: ", x_cont.grad.norm())
    # --- check that transformers are frozen ---------------------------