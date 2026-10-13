"""
exp06_pilot.py -- the feasibility gate for the inner-rank design.

Run:  python3.10 -u exp06_pilot.py | tee exp06.txt     (about 8 minutes)

Every number in exp05 used the ORACLE (mu, Sigma) to rank.  In practice the
ranking must use a plug-in Mahalanobis distance from a pilot (mu0, Sigma0).
This matters far more here than in balanced RSS, because the inner-rank design
concentrates the sample around mu0: if mu0 is wrong, the failure mode is a
BIAS, and bias does not shrink with n.

In the limit m -> infinity the selected unit is the point closest to mu0 among
m draws, so E[muhat] -> mu0 and the bias tends to the full pilot error.  There
is therefore a tension the balanced design does not have: larger m buys
variance reduction and contamination exclusion, but amplifies pilot error.
This script measures that trade-off.

VARIANTS
  oracle        rank by the true (mu, Sigma)                   [reference]
  madpilot      coordinatewise median + MAD diagonal scatter   [cheap, crude]
  mcdpilot      MCD on an independent pilot sample of size n_p [realistic]
  bias-b        mu0 = mu + b*e_1, true Sigma                   [controlled error]
  iterated-T    start crude, re-rank T times on own estimate   [self-correcting?]

Reported: RE against SRS at equal measured n, the bias norm, the mean number of
contaminated units admitted, and the corrected trace.
"""
import numpy as np, time, warnings
from scipy import stats, integrate
from sklearn.covariance import MinCovDet
warnings.filterwarnings("ignore")

SEED = 20260920


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def ED1(p, m):
    from math import lgamma, exp
    f = lambda x: stats.chi2.pdf(x, p); F = lambda x: stats.chi2.cdf(x, p)
    hi = stats.chi2.ppf(1 - 1e-12, p)
    c = exp(lgamma(m + 1) - lgamma(1) - lgamma(m))
    g = lambda x: x * c * (1 - F(x)) ** (m - 1) * f(x)
    return integrate.quad(g, 0, hi, limit=400)[0]


def draw(rng, shape, p, chol, eps, shift):
    n = int(np.prod(shape))
    X = rng.standard_normal((n, p)) @ chol.T
    f = rng.random(n) < eps
    k = int(f.sum())
    if k:
        d = np.zeros(p); d[0] = 1.0
        X[f] = shift * d + 0.5 * rng.standard_normal((k, p))
    return X.reshape(*shape, p), f.reshape(shape)


def pick_inner(U, F, mu0, S0inv):
    """From each set (axis 1) select the unit closest to mu0 in the S0 metric."""
    D = U - mu0
    d = np.einsum("...i,ij,...j->...", D, S0inv, D)
    o = np.argmin(d, axis=1)
    idx = np.arange(U.shape[0])
    return U[idx, o], F[idx, o]


def mad_pilot(X):
    mu0 = np.median(X, axis=0)
    s = 1.4826 * np.median(np.abs(X - mu0), axis=0)
    s = np.maximum(s, 1e-8)
    return mu0, np.diag(1.0 / s ** 2)


def run(p, m, k, reps, eps, variant, n_pilot=100, bias=0.0, iters=1, seed=0):
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    n = k * m
    mu = np.zeros(p)
    L_srs, L_in, B_in, N_in, T_in = [], [], [], [], []
    for _ in range(reps):
        U, F = draw(rng, (n, m), p, chol, eps, 6.0)
        Xs, _ = draw(rng, (n,), p, chol, eps, 6.0)
        L_srs.append(np.sum(Xs.mean(0) ** 2))

        if variant == "oracle":
            mu0, S0inv = mu, Sinv
        elif variant == "bias":
            mu0 = np.zeros(p); mu0[0] = bias; S0inv = Sinv
        elif variant == "madpilot":
            flat = U.reshape(-1, p)[: n]          # a cheap SRS-like pilot
            mu0, S0inv = mad_pilot(flat)
        elif variant == "mcdpilot":
            P, _ = draw(rng, (n_pilot,), p, chol, eps, 6.0)
            try:
                mc = MinCovDet(random_state=0).fit(P)
                mu0, S0inv = mc.location_, np.linalg.inv(mc.covariance_)
            except Exception:
                mu0, S0inv = mad_pilot(P)
        elif variant == "iterated":
            flat = U.reshape(-1, p)[: n]
            mu0, S0inv = mad_pilot(flat)
            for _t in range(iters):
                Xi, _ = pick_inner(U, F, mu0, S0inv)
                mu0 = Xi.mean(0)
                C = np.cov(Xi, rowvar=False) * (p / ED1(p, m))
                S0inv = np.linalg.inv(C + 1e-9 * np.eye(p))
        Xi, Fi = pick_inner(U, F, mu0, S0inv)
        mi = Xi.mean(0)
        L_in.append(np.sum(mi ** 2)); B_in.append(mi)
        N_in.append(int(Fi.sum()))
        T_in.append(np.trace(np.cov(Xi, rowvar=False)))
    L_srs = np.array(L_srs); L_in = np.array(L_in); B_in = np.array(B_in)
    fac = p / ED1(p, m)
    return {"RE": L_srs.mean() / L_in.mean(),
            "bias": np.linalg.norm(B_in.mean(0)),
            "N": np.mean(N_in), "trC": np.mean(T_in) * fac,
            "rmse": np.sqrt(L_in.mean())}


if __name__ == "__main__":
    t0 = time.time(); p, k = 4, 10
    print("p=4, Sigma=AR(1,0.5), tr(Sigma)=4.  RE is against SRS at equal measured n.\n")

    for eps in (0.0, 0.10, 0.20):
        print(f"================ eps = {eps:.2f} ================")
        print(f"{'variant':>14} {'m':>3} {'n':>4} {'RE':>7} {'bias':>7} "
              f"{'RMSE':>7} {'mean N':>7} {'corr tr':>8}")
        for m in (5, 10):
            for k, (tag, kw) in enumerate((
                    ("oracle", dict(variant="oracle")),
                    ("MAD pilot", dict(variant="madpilot")),
                    ("MCD pilot n=100", dict(variant="mcdpilot", n_pilot=100)),
                    ("iterated x1", dict(variant="iterated", iters=1)),
                    ("iterated x3", dict(variant="iterated", iters=3)))):
                # deterministic seed: hash() of a str is randomised per process
                r = run(p, m, max(1, 50 // m), 1200, eps,
                        seed=20000 + 1000 * m + 100 * int(round(100 * eps)) + k,
                        **kw)
                print(f"{tag:>14} {m:>3} {max(1,50//m)*m:>4} {r['RE']:7.3f} "
                      f"{r['bias']:7.4f} {r['rmse']:7.4f} {r['N']:7.2f} {r['trC']:8.3f}")
        print()

    print("================ controlled pilot error (clean data) ================")
    print("mu0 = mu + b*e_1 with the true Sigma; how much error can it absorb?\n")
    print(f"{'b':>6} {'m':>3} {'RE':>7} {'bias':>7} {'RMSE':>7}")
    for b in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        for m in (5, 10):
            r = run(p, m, max(1, 50 // m), 2000, 0.0, "bias", bias=b,
                    seed=int(1000 * b) + m)
            print(f"{b:6.2f} {m:>3} {r['RE']:7.3f} {r['bias']:7.4f} {r['rmse']:7.4f}")
    print(f"\nelapsed {time.time()-t0:.0f}s")
