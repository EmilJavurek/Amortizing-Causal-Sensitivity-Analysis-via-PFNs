"""Two-sided (corrected binary) KL CAPO bound for a known 1-D outcome law, without any flow optimizer.

Problem (upper head), over densities r = d nu / d phi on an equal-mass discretization y_k (p_k = 1/K)
of the exact observational law phi = P(Y | x, A=a):

    max_r  sum_k p_k r_k y_k   s.t.  sum p r = 1,  KL(nu||phi) = sum p r log r <= G,
                                     KL(phi||nu) = -sum p log r <= G,   r > 0.

Linear objective, convex constraints -> Lagrangian duality is exact. The dual function is
    g(alpha, beta, c) = c + (alpha + beta) G + sum_k p_k sup_r [ r (y_k - c) - alpha r log r + beta log r ].
The inner sup has the unique root s = log r of  alpha (s + 1) - beta e^{-s} = y_k - c  (increasing in s),
c is fixed by sum p r = 1 (decreasing in c), and (alpha, beta) >= 0 minimize the convex g with
envelope gradients dg/dalpha = G - sum p r log r, dg/dbeta = G + sum p log r. The lower head is
-(upper head of -y). The CAPO bound is pi * E_phi[Y] + (1 - pi) * E_nu[Y].

Why the outcome law suffices: for a fixed (x, a), condition any latent density ratio on
Y = f(x, a, U). Conditional Jensen inequalities weakly reduce both KL divergences while
preserving the outcome mean. Conversely, any outcome-law ratio r(Y) lifts to r(f(x, a, U))
with the same mean and divergences. No invertibility of the outcome map is required."""


from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


# Kl reference


ITER = 60
S_BRACKET = 40.0          # inner log-r bracket; a root within 1e-3 of it is treated as clipped -> failure
C_PAD = 50.0              # normalizer bracket around [min y, max y] (standardized units)
TOL_STATIONARY = 1e-8     # |alpha (s+1) - beta e^{-s} - (y - c)| at the returned inner roots
TOL_NORMAL = 1e-8         # |mean(r) - 1| before renormalization
TOL_FEASIBLE = 1e-5       # relative excess of either KL over Gamma
TOL_GAP = 1e-6            # dual value - primal value (standardized units, both certificates valid)

def _inner(y, c, alpha, beta):
    """Vectorized bisection for s = log r solving alpha (s + 1) - beta exp(-s) = y - c."""
    t = y - c
    lo = np.full_like(y, -S_BRACKET)
    hi = np.full_like(y, S_BRACKET)
    for _ in range(ITER):
        mid = 0.5 * (lo + hi)
        f = alpha * (mid + 1.0) - beta * np.exp(-mid) - t
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f >= 0, mid, hi)
    return 0.5 * (lo + hi)

def _solve_c(y, alpha, beta):
    lo, hi = y.min() - C_PAD, y.max() + C_PAD
    for _ in range(ITER):
        c = 0.5 * (lo + hi)
        mass = np.exp(_inner(y, c, alpha, beta)).mean()
        if mass > 1.0:
            lo = c
        else:
            hi = c
    c = 0.5 * (lo + hi)
    return c, _inner(y, c, alpha, beta)

def certify_inner(y, c, s, alpha, beta, z_lo, z_hi):
    """Fail loudly unless the inner sup is a genuine interior stationary point and c is interior."""
    assert np.all(np.abs(s) < S_BRACKET - 1e-3), 'inner log-r root clipped at the bracket'
    resid = alpha * (s + 1.0) - beta * np.exp(-s) - (y - c)
    assert np.max(np.abs(resid)) < TOL_STATIONARY * max(1.0, np.abs(y - c).max()), ('inner stationarity', np.abs(resid).max())
    assert z_lo - C_PAD + 1e-6 < c < z_hi + C_PAD - 1e-6, 'normalizer c clipped at the bracket'
    normal = abs(np.exp(s).mean() - 1.0)
    assert normal < TOL_NORMAL, ('normalization residual', normal)
    return dict(max_stationarity_residual=float(np.abs(resid).max()), normalization_residual=float(normal))

def _dual(params, y, gamma, certify=False):
    alpha, beta = np.exp(params)
    c, s = _solve_c(y, alpha, beta)
    cert = certify_inner(y, c, s, alpha, beta, y.min(), y.max()) if certify else None
    r = np.exp(s)
    r = r / r.mean()                       # remove residual normalization error
    s = np.log(r)
    value = c + (alpha + beta) * gamma + np.mean(r * (y - c) - alpha * r * s + beta * s)
    grad = np.array([alpha * (gamma - np.mean(r * s)), beta * (gamma + np.mean(s))])
    return (value, grad, r, cert) if certify else (value, grad, r)

def nu_mean_upper(y, gamma, start=(0.0, 0.0)):
    """max E_nu[Y] under max(KL(nu||phi), KL(phi||nu)) <= gamma; phi uniform on atoms y. Returns (value, r, params)."""
    y = np.asarray(y, dtype=float)
    assert gamma > 0
    scale = y.std() if y.std() > 0 else 1.0
    z = (y - y.mean()) / scale             # scale-free problem; mean is equivariant
    res = minimize(lambda p: _dual(p, z, gamma)[:2], x0=np.asarray(start, float), jac=True, method='L-BFGS-B',
                   bounds=[(-25.0, 8.0), (-25.0, 8.0)], options=dict(maxiter=200, gtol=1e-10, ftol=1e-14))
    dual_value, _, r, cert = _dual(res.x, z, gamma, certify=True)
    kl_fwd, kl_rev = float(np.mean(r * np.log(r))), float(-np.mean(np.log(r)))
    primal = float(np.mean(r * z))
    # Enforced certificates: primal feasibility, valid dual bound (interior inner sup), small gap and
    # complementary slackness. Any violation raises: no reference is written from a failed solve.
    assert kl_fwd <= gamma * (1 + TOL_FEASIBLE) and kl_rev <= gamma * (1 + TOL_FEASIBLE), ('infeasible', kl_fwd, kl_rev, gamma)
    gap = float(dual_value - primal)
    assert -TOL_GAP < gap < TOL_GAP, ('duality gap', gap, gamma)
    alpha, beta = np.exp(res.x)
    slack = max(alpha * (gamma - kl_fwd), beta * (gamma - kl_rev))
    assert slack < 1e-5, ('complementary slackness', slack)
    return y.mean() + scale * primal, dict(kl_fwd=kl_fwd, kl_rev=kl_rev, dual_gap=gap * scale, params=res.x,
                                           complementary_slackness=float(slack), **cert)

def capo_bounds(draws, pi, gammas, atoms=512):
    """CAPO KL bounds per row from exact-law draws (rows, n): equal-mass atoms = block means of sorted draws."""
    draws = np.sort(np.asarray(draws, dtype=float), axis=1)
    rows, n = draws.shape
    assert n % atoms == 0
    y_atoms = draws.reshape(rows, atoms, n // atoms).mean(axis=2)
    out_lo = np.empty((rows, len(gammas)))
    out_up = np.empty_like(out_lo)
    worst_gap = 0.0
    for i in range(rows):
        mu = draws[i].mean()
        su, sl = (0.0, 0.0), (0.0, 0.0)
        for j, g in enumerate(gammas):
            up, iu = nu_mean_upper(y_atoms[i], g, su)
            lo_neg, il = nu_mean_upper(-y_atoms[i], g, sl)
            su, sl = iu['params'], il['params']
            worst_gap = max(worst_gap, abs(iu['dual_gap']), abs(il['dual_gap']))
            out_up[i, j] = pi[i] * mu + (1 - pi[i]) * up
            out_lo[i, j] = pi[i] * mu + (1 - pi[i]) * (-lo_neg)
    return out_lo, out_up, worst_gap
