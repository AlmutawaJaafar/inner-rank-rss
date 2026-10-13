"""
exp07_poolpilot.py

QUESTION.  exp06 showed that the inner-rank design loses its clean-data
advantage under a crude pilot (RE 0.85 with a MAD pilot fitted on n = 50
units) because the estimator's bias is c(m) times the pilot's error, with
c(5) ~ 0.64.  Break-even sits at a pilot error of about 0.35, roughly one
standard error of the SRS mean.

But under Model B the whole screened pool of n*m units is already measured, so
the pilot need not be fitted on n units -- it can use all n*m.  With n = 50,
m = 5 that is 250 units and a pilot standard error near 0.13, well inside the
break-even.  This script asks whether that restores the gain.

VARIANTS
  oracle          true (mu, Sigma)                                [ceiling]
  pilot-n         MAD pilot on n units              (exp06 baseline, failed)
  pool-MAD        MAD pilot on all n*m screened units
  pool-MCD        MCD pilot on all n*m screened units
  split-MCD       MCD pilot on an INDEPENDENT half of the sets, selection on
                  the other half -- removes the dependence between the ranking
                  function and the selected units, which the pool variants
                  induce and which the finite-sample theory needs.

The split variant is the one that matters for the theory: if the pilot is
fitted on the same units that are then ranked and selected, the selected
observations are no longer independent of the ranking and the exact
finite-sample results no longer apply.  Sample splitting restores them at the
cost of half the sets.

Run:  python3.10 -u exp07_poolpilot.py | tee exp07.txt      (about 10 minutes)
"""
import numpy as np, time, warnings
from scipy import stats, integrate
from sklearn.covariance import MinCovDet
warnings.filterwarnings("ignore")

SEED = 20260921


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def ED1(p, m):
    from math import lgamma, exp
    f = lambda x: stats.chi2.pdf(x, p); F = lambda x: stats.chi2.cdf(x, p)
    hi = stats.chi2.ppf(1 - 1e-12, p)
    c = exp(lgamma(m + 1) - lgamma(1) - lgamma(m))
    return integrate.quad(lambda x: x * c * (1 - F(x)) ** (m - 1) * f(x),
                          0, hi, limit=400)[0]


def draw(rng, shape, p, chol, eps, shift):
    n = int(np.prod(shape))
    X = rng.standard_normal((n, p)) @ chol.T
    f = rng.random(n) < eps
    k = int(f.sum())
    if k:
        d = np.zeros(p); d[0] = 1.0
        X[f] = shift * d + 0.5 * rng.standard_normal((k, p))
    return X.reshape(*shape, p), f.reshape(shape)


def mad_pilot(X):
    mu0 = np.median(X, axis=0)
    s = np.maximum(1.4826 * np.median(np.abs(X - mu0), axis=0), 1e-8)
    return mu0, np.diag(1.0 / s ** 2)


def mcd_pilot(X):
    try:
        mc = MinCovDet(random_state=0).fit(X)
        return mc.location_, np.linalg.inv(mc.covariance_)
    except Exception:
        return mad_pilot(X)


def pick_inner(U, F, mu0, S0inv):
    D = U - mu0
    d = np.einsum("...i,ij,...j->...", D, S0inv, D)
    o = np.argmin(d, axis=1)
    i = np.arange(U.shape[0])
    return U[i, o], F[i, o]


def run(p, m, n, reps, eps, variant, seed=0):
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    mu = np.zeros(p); fac = p / ED1(p, m)
    Ls, Li, Bi, Ni, Ti, Pe = [], [], [], [], [], []
    for _ in range(reps):
        U, F = draw(rng, (n, m), p, chol, eps, 6.0)     # the screened pool
        Xs, _ = draw(rng, (n,), p, chol, eps, 6.0)      # SRS comparator
        Ls.append(np.sum(Xs.mean(0) ** 2))
        pool = U.reshape(-1, p)

        if variant == "oracle":
            mu0, S0inv = mu, Sinv; Us, Fs = U, F
        elif variant == "pilot-n":
            mu0, S0inv = mad_pilot(pool[:n]); Us, Fs = U, F
        elif variant == "pool-MAD":
            mu0, S0inv = mad_pilot(pool); Us, Fs = U, F
        elif variant == "pool-MCD":
            mu0, S0inv = mcd_pilot(pool); Us, Fs = U, F
        elif variant == "split-MCD":
            h = n // 2
            mu0, S0inv = mcd_pilot(U[:h].reshape(-1, p))
            Us, Fs = U[h:], F[h:]                       # disjoint sets
        Pe.append(np.linalg.norm(mu0 - mu))
        Xi, Fi = pick_inner(Us, Fs, mu0, S0inv)
        mi = Xi.mean(0)
        Li.append(np.sum(mi ** 2)); Bi.append(mi)
        Ni.append(int(Fi.sum())); Ti.append(np.trace(np.cov(Xi, rowvar=False)))
    Ls = np.array(Ls); Li = np.array(Li); Bi = np.array(Bi)
    B = len(Ls)
    re_relse = np.sqrt(Ls.var(ddof=1) / (B * Ls.mean() ** 2) +
                       Li.var(ddof=1) / (B * Li.mean() ** 2))
    return {"re_relse": re_relse, "pilot_rms": float(np.sqrt(np.mean(np.square(Pe)))),
            "RE": Ls.mean() / Li.mean(), "bias": np.linalg.norm(Bi.mean(0)),
            "rmse": np.sqrt(Li.mean()), "N": np.mean(Ni),
            "trC": np.mean(Ti) * fac, "pilot_err": np.mean(Pe),
            "n_used": Xi.shape[0]}


if __name__ == "__main__":
    t0 = time.time(); p = 4
    print("p=4, Sigma=AR(1,0.5).  RE is against an SRS of n measured units.")
    print("'pilot err' is the mean ||mu0 - mu||; break-even is near 0.35 at m=5.\n")
    for eps in (0.00, 0.10, 0.20):
        print(f"================= eps = {eps:.2f} =================")
        print(f"{'variant':>11} {'m':>3} {'n used':>7} {'RE':>7} {'bias':>7} "
              f"{'RMSE':>7} {'pilot err':>10} {'mean N':>7} {'corr tr':>8}")
        for m in (5, 10):
            n = max(1, 50 // m) * m
            for k, v in enumerate(("oracle", "pilot-n", "pool-MAD", "pool-MCD",
                                   "split-MCD")):
                # deterministic seed (hash() of a str is randomised per process)
                r = run(p, m, n, 1200, eps, v,
                        seed=10000 + 1000 * m + 100 * int(round(100 * eps)) + k)
                print(f"{v:>11} {m:>3} {r['n_used']:>7} {r['RE']:7.3f} "
                      f"{r['bias']:7.4f} {r['rmse']:7.4f} {r['pilot_err']:10.4f} "
                      f"{r['N']:7.2f} {r['trC']:8.3f}"
                      f"   [rel.se(RE) {r['re_relse']:.3f}, rms pilot err {r['pilot_rms']:.4f}]")
        print()
    print(f"elapsed {time.time()-t0:.0f}s")
