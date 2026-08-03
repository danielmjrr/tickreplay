"""Bayesian online changepoint detection (Adams & MacKay 2007), Student-t predictive.

Strictly filtered: the run-length posterior at index i uses observations <= i only.
Normal-Inverse-Gamma conjugate observation model; constant hazard; run-length vector
truncated for speed (mass < trunc dropped, hard cap on vector size).
"""
import numpy as np
from scipy.special import gammaln


def _t_logpdf(x, df, loc, scale):
    z = (x - loc) / scale
    return (gammaln((df + 1) / 2) - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi) - np.log(scale)
            - (df + 1) / 2 * np.log1p(z * z / df))


def bocpd_features(x, hazard=1 / 250, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0,
                   cap=3000, trunc=1e-6):
    """Returns per-observation filtered {"p_cp_10": P(run<=10), "erl": E[run length]}."""
    x = np.asarray(x, dtype="float64")
    n = len(x)
    p_cp = np.zeros(n)
    erl = np.zeros(n)
    rl = np.array([0])                      # run lengths carried by each hypothesis
    R = np.array([1.0])
    mu, kappa = np.array([mu0]), np.array([kappa0])
    alpha, beta = np.array([alpha0]), np.array([beta0])
    for i in range(n):
        pred = np.exp(_t_logpdf(x[i], 2 * alpha, mu,
                                np.sqrt(beta * (kappa + 1) / (alpha * kappa))))
        growth = R * pred * (1 - hazard)
        cp = float((R * pred).sum() * hazard)
        R = np.r_[cp, growth]
        R /= R.sum()
        rl = np.r_[0, rl + 1]
        beta = np.r_[beta0, beta + kappa * (x[i] - mu) ** 2 / (2 * (kappa + 1))]
        mu = np.r_[mu0, (kappa * mu + x[i]) / (kappa + 1)]
        alpha = np.r_[alpha0, alpha + 0.5]
        kappa = np.r_[kappa0, kappa + 1]
        p_cp[i] = R[rl <= 10].sum()
        erl[i] = (rl * R).sum()
        keep = R >= trunc
        keep[0] = True
        if keep.sum() > cap:
            keep &= R >= np.partition(R, -cap)[-cap]
        rl, R = rl[keep], R[keep]
        R /= R.sum()
        mu, kappa, alpha, beta = mu[keep], kappa[keep], alpha[keep], beta[keep]
    return {"p_cp_10": p_cp, "erl": erl}
