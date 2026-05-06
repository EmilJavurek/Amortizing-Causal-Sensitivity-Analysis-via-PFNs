# training_standard.py — PI Foundation Model trainer
# Replaces CATETrainer. Everything above the class definition
# (imports, set_seed, format_time, parse_args) is additive or
# replaces old references; the class itself is new.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
from typing import Dict, List, Tuple
import logging
import time


from src.tabpfn.model.causalFM import PerFeatureTransformerPIFM
from Setting_standard.causaldataset import create_pifm_data_loaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOUND_UPPER = 0
BOUND_LOWER = 1


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.2f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.2f}s"


def parse_args():
    parser = argparse.ArgumentParser(description='Train PI Foundation Model')

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument('--data_path', type=str,
                        default="Setting_standard/training_data")
    parser.add_argument('--val_split',    type=float, default=0.2)
    parser.add_argument('--m_prime',      type=int,   default=64,
                        help='Query IDs subsampled per DGP per step')
    parser.add_argument('--g',            type=int,   default=10,
                        help='Gamma points sampled per query ID')

    # ── Optimisation ──────────────────────────────────────────────────────
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs',       type=int,   default=150)
    parser.add_argument('--early_stop',   type=int,   default=50)
    parser.add_argument('--clip_grad',    type=float, default=1.0)

    # ── Monotonicity penalty ──────────────────────────────────────────────
    parser.add_argument('--mono_weight',  type=float, default=1.0,
                        help='Weight on monotonicity penalty. 0.0 disables it.')

    # ── Architecture ─────────────────────────────────────────────────────
    parser.add_argument('--gmm_n_components', type=int,   default=5)
    parser.add_argument('--gmm_min_sigma',    type=float, default=1e-3)
    parser.add_argument('--gmm_pi_temp',      type=float, default=1.0)

    # ── Infrastructure ────────────────────────────────────────────────────
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--save_freq',   type=int,   default=10)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--save_dir',    type=str,   default='checkpoints')
    parser.add_argument('--log_dir',     type=str,   default='logs')
    parser.add_argument('--gpu',         type=int,   default=0)

    return parser.parse_args()


# ── GMM helpers ───────────────────────────────────────────────────────────────

def gmm_nll_loss(pi, mu, sigma, target, eps=1e-12):
    """
    Negative log-likelihood of target under a Gaussian mixture model.

    Args:
        pi:     (N, K)  mixture weights
        mu:     (N, K)  component means
        sigma:  (N, K)  component standard deviations
        target: (N,)    scalar targets

    Returns:
        scalar mean NLL
    """
    assert pi.shape == mu.shape == sigma.shape
    assert target.shape == (pi.shape[0],), \
        f"target shape {target.shape} inconsistent with pi shape {pi.shape}"

    pi    = torch.clamp(pi, min=eps)
    pi    = pi / pi.sum(dim=-1, keepdim=True)
    sigma = torch.clamp(sigma, min=eps)

    z = (target.unsqueeze(-1) - mu) / sigma           # (N, K)
    log_prob = -0.5 * z * z - torch.log(sigma) - 0.5 * np.log(2.0 * np.pi)
    log_mix  = torch.logsumexp(torch.log(pi) + log_prob, dim=-1)  # (N,)
    return -log_mix.mean()


def sample_gmm(pi, mu, sigma, n_samples: int = 1000) -> torch.Tensor:
    """
    Draw samples from a batch of GMMs for calibration estimation.

    Args:
        pi, mu, sigma: (N, K)

    Returns:
        samples: (N, n_samples)

    NOTE: inputs must be on CPU. torch.multinomial has precision issues on
    CUDA for large N. compute_calibration_diagnostics calls this with .cpu()
    tensors — do not call on GPU tensors directly.
    """
    N, K = pi.shape
    # Sample component indices
    comp = torch.multinomial(pi, n_samples, replacement=True)          # (N, n_samples)
    # Gather selected means and sigmas
    mu_s    = mu.gather(1, comp)    # (N, n_samples)  — gather along K dim
    sigma_s = sigma.gather(1, comp) # (N, n_samples)
    return mu_s + sigma_s * torch.randn_like(mu_s)


# ── Sequence construction ─────────────────────────────────────────────────────

def build_pifm_sequence(batch: Dict, device: torch.device, n: int):
    """
    Assemble context and query tensors from a collated batch into the
    full (seq_len, B, ...) format expected by PerFeatureTransformerPIFM.forward().

    Context rows carry real (X, a, y) and NaN gamma.
    Query rows carry real (x_query, a_query, gamma) and NaN y.

    Args:
        batch:  output of collate_pifm
        device: target device
        n:      number of context rows (= dataset size per DGP)

    Returns:
        x_dict, a_dict, y_dict, gamma_dict: each {"main": Tensor(n+m, B, ...)}
        single_eval_pos: int = n
    """
    _EXPECTED_BATCH_KEYS = {"X", "a_context", "y_context", "x_query", "a_query", "gamma"}
    assert _EXPECTED_BATCH_KEYS.issubset(batch.keys()), \
        f"batch missing keys: {_EXPECTED_BATCH_KEYS - batch.keys()}"
    
    def _d(t):
        return t.to(device=device, dtype=torch.float32)

    X         = _d(batch["X"])           # (n, B, d_x)
    a_context = _d(batch["a_context"])   # (n, B, 1)
    y_context = _d(batch["y_context"])   # (n, B, 1)
    x_query   = _d(batch["x_query"])     # (m, B, d_x)
    a_query   = _d(batch["a_query"])     # (m, B, 1)
    gamma_q   = _d(batch["gamma"])       # (m, B, 1)

    m = x_query.shape[0]
    B = X.shape[1]

    def _nan(shape):
        return torch.full(shape, float("nan"), device=device, dtype=torch.float32)

    x_full     = torch.cat([X,         x_query],          dim=0)  # (n+m, B, d_x)
    a_full     = torch.cat([a_context, a_query],           dim=0)  # (n+m, B, 1)
    y_full     = torch.cat([y_context, _nan((m, B, 1))],   dim=0)  # (n+m, B, 1)
    gamma_full = torch.cat([_nan((n, B, 1)), gamma_q],     dim=0)  # (n+m, B, 1)

    return (
        {"main": x_full},
        {"main": a_full},
        {"main": y_full},
        {"main": gamma_full},
        n,
    )


# ── Monotonicity penalty ──────────────────────────────────────────────────────

def monotonicity_penalty(
    upper_mean: torch.Tensor,
    lower_mean: torch.Tensor,
    gamma_bm:   torch.Tensor,
    m_prime:    int,
    g:          int,
) -> Tuple[torch.Tensor, Dict]:
    """
    Soft monotonicity penalty over gamma-sorted predictions within each
    (DGP, query_id) group.

    Upper bound must be non-decreasing in Gamma.
    Lower bound must be non-increasing in Gamma.

    This function relies on the block structure preserved by PIFMDataset
    (shuffle removed): rows [i*g : (i+1)*g] within each DGP all share
    the same query_id.

    Args:
        upper_mean: (B, m)   GMM mean predictions for upper head at query positions
        lower_mean: (B, m)   GMM mean predictions for lower head at query positions
        gamma_bm:   (B, m)   gamma values at query positions
        m_prime:    int      number of query groups per DGP
        g:          int      gamma points per group

    Returns:
        penalty:  scalar differentiable penalty
        diagnostics: dict with non-differentiable diagnostic scalars
    """
    B, m = upper_mean.shape
    assert m == m_prime * g, \
        f"Expected m={m_prime * g} from m_prime*g, got {m}"

    # Reshape to (B, m_prime, g)
    upper_g = upper_mean.reshape(B, m_prime, g)    # (B, m', g)
    lower_g = lower_mean.reshape(B, m_prime, g)
    gamma_g = gamma_bm.reshape(B, m_prime, g)

    # Sort all three by gamma within each (B, m') group
    gamma_sorted, sort_idx = gamma_g.sort(dim=-1)  # (B, m', g)
    upper_sorted = upper_g.gather(2, sort_idx)
    lower_sorted = lower_g.gather(2, sort_idx)

    # Adjacent differences after gamma-sorting
    # upper should be non-decreasing → diffs should be >= 0
    upper_diffs = upper_sorted[:, :, 1:] - upper_sorted[:, :, :-1]  # (B, m', g-1)
    # lower should be non-increasing → diffs should be <= 0
    lower_diffs = lower_sorted[:, :, 1:] - lower_sorted[:, :, :-1]  # (B, m', g-1)

    upper_violations = torch.relu(-upper_diffs)  # > 0 when upper decreases
    lower_violations = torch.relu(lower_diffs)   # > 0 when lower increases

    penalty = upper_violations.mean() + lower_violations.mean()

    # ── Non-differentiable diagnostics (detached) ─────────────────────────
    with torch.no_grad():
        viol_frac_upper = (upper_diffs < 0).float().mean().item()
        viol_frac_lower = (lower_diffs > 0).float().mean().item()

        u_viol_mask = upper_diffs < 0
        l_viol_mask = lower_diffs > 0
        viol_mag_upper = upper_diffs[u_viol_mask].abs().mean().item() \
            if u_viol_mask.any() else 0.0
        viol_mag_lower = lower_diffs[l_viol_mask].abs().mean().item() \
            if l_viol_mask.any() else 0.0

    diagnostics = {
        "mono/violation_frac_upper":    viol_frac_upper,
        "mono/violation_frac_lower":    viol_frac_lower,
        "mono/violation_magnitude_upper": viol_mag_upper,
        "mono/violation_magnitude_lower": viol_mag_lower,
        "mono/penalty":                 penalty.item(),
    }

    return penalty, diagnostics


# ── Calibration diagnostics (validation only) ─────────────────────────────────

@torch.no_grad()
def compute_calibration_diagnostics(
    pi:         torch.Tensor,
    mu:         torch.Tensor,
    sigma:      torch.Tensor,
    theta_star: torch.Tensor,
    gamma_vals: torch.Tensor,
    prefix:     str,
    n_samples:  int = 2000,
) -> Dict[str, float]:
    """
    Compute calibration and bound-quality diagnostics for one head.

    Args:
        pi, mu, sigma:  (N, K)  GMM parameters
        theta_star:     (N,)    ground-truth bound values
        gamma_vals:     (N,)    gamma values (for stratification)
        prefix:         "upper" or "lower"
        n_samples:      Monte Carlo samples for interval estimation

    Returns:
        dict of {metric_name: scalar}
    """
    assert pi.shape[0] == theta_star.shape[0] == gamma_vals.shape[0]

    pred_mean = (pi * mu).sum(dim=-1)          # (N,)
    residuals = pred_mean - theta_star          # (N,)
    bias = residuals.mean().item()
    rmse = residuals.pow(2).mean().sqrt().item()

    # Monte Carlo credible intervals
    samples   = sample_gmm(pi, mu, sigma, n_samples=n_samples)  # (N, n_samples)
    q05       = torch.quantile(samples, 0.05, dim=1)  # (N,)
    q25       = torch.quantile(samples, 0.25, dim=1)
    q75       = torch.quantile(samples, 0.75, dim=1)
    q95       = torch.quantile(samples, 0.95, dim=1)

    covered_90 = ((theta_star >= q05) & (theta_star <= q95)).float().mean().item()
    covered_50 = ((theta_star >= q25) & (theta_star <= q75)).float().mean().item()
    width_90   = (q95 - q05).mean().item()

    # Stratify by gamma quartile
    q25_gamma = torch.quantile(gamma_vals, 0.25)
    low_gamma_mask  = gamma_vals <= q25_gamma
    high_gamma_mask = gamma_vals >= torch.quantile(gamma_vals, 0.75)

    rmse_low_gamma  = residuals[low_gamma_mask].pow(2).mean().sqrt().item() \
        if low_gamma_mask.any() else float("nan")
    rmse_high_gamma = residuals[high_gamma_mask].pow(2).mean().sqrt().item() \
        if high_gamma_mask.any() else float("nan")

    return {
        f"bounds/{prefix}_bias":              bias,
        f"bounds/{prefix}_rmse":              rmse,
        f"calib/{prefix}_coverage_90":        covered_90,
        f"calib/{prefix}_coverage_50":        covered_50,
        f"calib/{prefix}_interval_width_90":  width_90,
        f"bounds/{prefix}_rmse_low_gamma":    rmse_low_gamma,
        f"bounds/{prefix}_rmse_high_gamma":   rmse_high_gamma,
    }


# ── Trainer ───────────────────────────────────────────────────────────────────

class PIFMTrainer:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
        )
        self.m_prime = args.m_prime
        self.g       = args.g
        self.m       = args.m_prime * args.g     # total query rows per DGP per step

        self.model = PerFeatureTransformerPIFM(
            gmm_n_components=args.gmm_n_components,
            gmm_min_sigma=args.gmm_min_sigma,
            gmm_pi_temp=args.gmm_pi_temp,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.train_loader, self.val_loader = create_pifm_data_loaders(
            root_dir=args.data_path,
            m_prime=args.m_prime,
            g=args.g,
            batch_size=args.batch_size,
            val_split=args.val_split,
            shuffle=True,
            num_workers=args.num_workers,
            log_fn=logger.info,
        )

        self.writer = SummaryWriter(log_dir=args.log_dir)

        # Scalars tracked across epochs
        self.train_losses:   List[float] = []
        self.val_losses:     List[float] = []
        self.best_val_loss:  float       = float("inf")
        self.epoch_times:    List[float] = []
        self.validation_times: List[float] = []
        self.total_training_time: float  = 0.0


    # ── Loss ──────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        out:        Dict,
        batch:      Dict,
        n:          int,
        step_diagnostics: Dict,
    ) -> torch.Tensor:
        """
        Compute total loss = NLL_upper + NLL_lower + mono_weight * mono_penalty.

        Modifies step_diagnostics in-place with per-step scalars.
        Returns differentiable total loss scalar.
        """
        B   = out["gmm_upper_pi"].shape[0]
        m   = self.m

        # Extract query-position predictions: (B, m, K)
        pi_u  = out["gmm_upper_pi"][:,  n:, :]
        mu_u  = out["gmm_upper_mu"][:,  n:, :]
        sig_u = out["gmm_upper_sigma"][:, n:, :]
        pi_l  = out["gmm_lower_pi"][:,  n:, :]
        mu_l  = out["gmm_lower_mu"][:,  n:, :]
        sig_l = out["gmm_lower_sigma"][:, n:, :]

        # Targets: (m, B) → (B, m)
        bound_type = batch["bound_type"].to(self.device).permute(1, 0)   # (B, m)
        theta      = batch["theta_star"].to(self.device)\
                         .permute(1, 0, 2).squeeze(-1)                   # (B, m)

        # Flatten to (B*m, ...) for loss computation
        pi_u_f  = pi_u.reshape(B * m, -1)
        mu_u_f  = mu_u.reshape(B * m, -1)
        sig_u_f = sig_u.reshape(B * m, -1)
        pi_l_f  = pi_l.reshape(B * m, -1)
        mu_l_f  = mu_l.reshape(B * m, -1)
        sig_l_f = sig_l.reshape(B * m, -1)
        theta_f = theta.reshape(B * m)
        is_upper = (bound_type.reshape(B * m) == BOUND_UPPER)

        # NLL losses — each head trains only on its own bound type
        loss_upper = gmm_nll_loss(
            pi_u_f[is_upper], mu_u_f[is_upper], sig_u_f[is_upper], theta_f[is_upper]
        ) if is_upper.any() else torch.tensor(0.0, device=self.device)

        loss_lower = gmm_nll_loss(
            pi_l_f[~is_upper], mu_l_f[~is_upper], sig_l_f[~is_upper], theta_f[~is_upper]
        ) if (~is_upper).any() else torch.tensor(0.0, device=self.device)

        # Monotonicity penalty
        upper_mean = (pi_u * mu_u).sum(dim=-1)   # (B, m)
        lower_mean = (pi_l * mu_l).sum(dim=-1)   # (B, m)
        gamma_bm   = batch["gamma"].to(self.device)\
                         .permute(1, 0, 2).squeeze(-1)                   # (B, m)

        mono_pen, mono_diag = monotonicity_penalty(
            upper_mean, lower_mean, gamma_bm,
            m_prime=self.m_prime, g=self.g,
        )

        total_loss = loss_upper + loss_lower + self.args.mono_weight * mono_pen

        # Record step-level diagnostics (detached scalars, not tensors)
        step_diagnostics.update({
            "loss/train_upper": loss_upper.item(),
            "loss/train_lower": loss_lower.item(),
            "loss/train_total": total_loss.item(),
            "data/upper_frac":  is_upper.float().mean().item(),
        })
        step_diagnostics.update(mono_diag)

        return total_loss

    # ── Training epoch ────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss  = 0.0
        n_batches   = 0
        epoch_start = time.time()

        # Running averages for step diagnostics logged to TB
        running_diag: Dict[str, float] = {}

        with tqdm(self.train_loader,
                  desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]") as pbar:
            for batch in pbar:
                n = batch["X"].shape[0]   # context rows (should equal dataset size)

                x_d, a_d, y_d, gamma_d, single_eval_pos = build_pifm_sequence(
                    batch, self.device, n
                )

                out = self.model(x_d, a_d, y_d, gamma_d,
                                 single_eval_pos=single_eval_pos)

                step_diag: Dict = {}
                loss = self._compute_loss(out, batch, n, step_diag)

                self.optimizer.zero_grad()
                loss.backward()
                if self.args.clip_grad > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad
                    )
                self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

                for k, v in step_diag.items():
                    running_diag[k] = running_diag.get(k, 0.0) + v

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mono": f"{step_diag.get('mono/penalty', 0.0):.4f}",
                })

        epoch_time = time.time() - epoch_start
        self.epoch_times.append(epoch_time)

        avg_loss = total_loss / max(n_batches, 1)
        self.train_losses.append(avg_loss)

        # Log epoch-averaged step diagnostics to TB
        for k, v in running_diag.items():
            self.writer.add_scalar(k, v / max(n_batches, 1), epoch)
        self.writer.add_scalar("Learning_rate",
                               self.optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch {epoch+1} train: loss={avg_loss:.4f}  "
            f"time={format_time(epoch_time)}"
        )

        # NOTE: TEMPORARY CHECK
        if torch.cuda.is_available():
            logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        return avg_loss

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss   = 0.0
        n_batches    = 0
        val_start    = time.time()

        # Accumulators for calibration diagnostics
        all_pi_u:    List[torch.Tensor] = []
        all_mu_u:    List[torch.Tensor] = []
        all_sig_u:   List[torch.Tensor] = []
        all_pi_l:    List[torch.Tensor] = []
        all_mu_l:    List[torch.Tensor] = []
        all_sig_l:   List[torch.Tensor] = []
        all_theta_u: List[torch.Tensor] = []
        all_theta_l: List[torch.Tensor] = []
        all_gamma_u: List[torch.Tensor] = []
        all_gamma_l: List[torch.Tensor] = []

        with torch.no_grad():
            with tqdm(self.val_loader,
                      desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]") as pbar:
                for batch in pbar:
                    n = batch["X"].shape[0]

                    x_d, a_d, y_d, gamma_d, single_eval_pos = build_pifm_sequence(
                        batch, self.device, n
                    )
                    out = self.model(x_d, a_d, y_d, gamma_d,
                                     single_eval_pos=single_eval_pos)

                    step_diag: Dict = {}
                    loss = self._compute_loss(out, batch, n, step_diag)

                    total_loss += loss.item()
                    n_batches  += 1
                    pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

                    # Collect per-head predictions and targets for calibration
                    pi_u  = out["gmm_upper_pi"][:,  n:, :].reshape(-1, out["gmm_upper_pi"].shape[-1])
                    mu_u  = out["gmm_upper_mu"][:,  n:, :].reshape(-1, out["gmm_upper_mu"].shape[-1])
                    sig_u = out["gmm_upper_sigma"][:, n:, :].reshape(-1, out["gmm_upper_sigma"].shape[-1])
                    pi_l  = out["gmm_lower_pi"][:,  n:, :].reshape(-1, out["gmm_lower_pi"].shape[-1])
                    mu_l  = out["gmm_lower_mu"][:,  n:, :].reshape(-1, out["gmm_lower_mu"].shape[-1])
                    sig_l = out["gmm_lower_sigma"][:, n:, :].reshape(-1, out["gmm_lower_sigma"].shape[-1])

                    bound_type_flat = batch["bound_type"].to(self.device)\
                                          .permute(1, 0).reshape(-1)     # (B*m,)
                    theta_flat      = batch["theta_star"].to(self.device)\
                                          .permute(1, 0, 2).squeeze(-1).reshape(-1)
                    gamma_flat      = batch["gamma"].to(self.device)\
                                          .permute(1, 0, 2).squeeze(-1).reshape(-1)

                    is_upper = (bound_type_flat == BOUND_UPPER)

                    all_pi_u.append(pi_u[is_upper].cpu())
                    all_mu_u.append(mu_u[is_upper].cpu())
                    all_sig_u.append(sig_u[is_upper].cpu())
                    all_theta_u.append(theta_flat[is_upper].cpu())
                    all_gamma_u.append(gamma_flat[is_upper].cpu())

                    all_pi_l.append(pi_l[~is_upper].cpu())
                    all_mu_l.append(mu_l[~is_upper].cpu())
                    all_sig_l.append(sig_l[~is_upper].cpu())
                    all_theta_l.append(theta_flat[~is_upper].cpu())
                    all_gamma_l.append(gamma_flat[~is_upper].cpu())

        val_time = time.time() - val_start
        self.validation_times.append(val_time)

        avg_loss = total_loss / max(n_batches, 1)
        self.val_losses.append(avg_loss)
        self.scheduler.step(avg_loss)

        # ── Calibration and bound-quality diagnostics ─────────────────────
        # NOTE: This scales with validation set size and may be expensive for large sets
        # Should be fine for our current sizes, but monitor and consider changing.
        self.writer.add_scalar("loss/val_total", avg_loss, epoch)

        def _cat(lst):
            return torch.cat(lst, dim=0) if lst else None

        pi_u_all  = _cat(all_pi_u);  mu_u_all = _cat(all_mu_u)
        sig_u_all = _cat(all_sig_u); th_u_all = _cat(all_theta_u)
        gam_u_all = _cat(all_gamma_u)
        pi_l_all  = _cat(all_pi_l);  mu_l_all = _cat(all_mu_l)
        sig_l_all = _cat(all_sig_l); th_l_all = _cat(all_theta_l)
        gam_l_all = _cat(all_gamma_l)

        if pi_u_all is not None and len(pi_u_all) > 0:
            diag_u = compute_calibration_diagnostics(
                pi_u_all, mu_u_all, sig_u_all, th_u_all, gam_u_all, prefix="upper"
            )
            for k, v in diag_u.items():
                self.writer.add_scalar(k, v, epoch)

        if pi_l_all is not None and len(pi_l_all) > 0:
            diag_l = compute_calibration_diagnostics(
                pi_l_all, mu_l_all, sig_l_all, th_l_all, gam_l_all, prefix="lower"
            )
            for k, v in diag_l.items():
                self.writer.add_scalar(k, v, epoch)

        logger.info(
            f"Epoch {epoch+1} val:   loss={avg_loss:.4f}  "
            f"time={format_time(val_time)}"
        )
        return avg_loss

    # ── Main loop ─────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss":             val_loss,
            "epoch_times":          self.epoch_times,
        }
        path = os.path.join(self.args.save_dir, f"model_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        if val_loss < self.best_val_loss:
            torch.save(checkpoint, os.path.join(self.args.save_dir, "best_model.pth"))
            logger.info(f"Best model saved (val_loss={val_loss:.4f})")

    def train(self) -> None:
        os.makedirs(self.args.save_dir, exist_ok=True)
        train_start      = time.time()
        early_stop_count = 0

        logger.info(f"Training on {self.device}")
        logger.info(f"Train DGPs: {len(self.train_loader.dataset)}  "
                    f"Val DGPs: {len(self.val_loader.dataset)}")
        logger.info(f"m_prime={self.m_prime}, g={self.g}, m={self.m}, "
                    f"B={self.args.batch_size}, seq_len≈{1024 + self.m}")

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            val_loss   = self.validate(epoch)

            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(epoch + 1, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.args.early_stop:
                logger.info(f"Early stopping after epoch {epoch+1}")
                break

        self.total_training_time = time.time() - train_start
        self.save_checkpoint(epoch + 1, val_loss)
        logger.info(f"Training complete in {format_time(self.total_training_time)}")
        self.writer.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.log_dir,  exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    trainer = PIFMTrainer(args)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()