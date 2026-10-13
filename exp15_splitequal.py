"""
exp15_splitequal.py -- what does sample splitting actually cost?

Run:  python3.10 -u exp15_splitequal.py | tee exp15.txt      (about 4 minutes)

Table `pilot` reports split-MCD measuring n/2 units against an SRS of n, and
the text then extrapolated to equal measured size by DOUBLING the relative
efficiency.  That extrapolation is wrong.  Writing the risk as

    R(n_meas) = A / n_meas + B,

where B is the contribution of pilot error and does not shrink with the number
of measured units, the relative efficiency against a fixed SRS of n is

    RE(n_meas) = (tr Sigma / n) / (A / n_meas + B),

so going from n/2 to n multiplies RE by (2A/n + B)/(A/n + B), which lies in
(1, 2] and equals 2 only when B = 0.  Doubling therefore assumes the pilot is
exact, which is precisely what the split variant is not.

This script measures the equal-size value instead of extrapolating to it.

  split-MCD        pilot on the first n/2 sets, measure the remaining n/2 sets
  split-MCD-equal  screen 3n/2 sets; pilot on the first n/2 sets (IDENTICAL
                   pilot sample size, so the pilot error is held fixed), then
                   measure the remaining n sets

The second is the exact thought experiment the text described, so the two rows
isolate the cost of measuring half as many units from the cost of splitting.
Both are compared against an SRS of n, as in Table `pilot`.
"""
import numpy as np, time, warnings
from scipy import stats, integrate
from sklearn.covariance import MinCovDet
warnings.filterwarnings("ignore")

SEED = 20260929


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
    """n = number of sets in the BASELINE design = SRS comparator size."""
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    mu = np.zeros(p); fac = p / ED1(p, m)
    h = n // 2                                   # sets used for the pilot
    n_screen = {"split-MCD": n, "split-MCD-equal": n + h,
                "oracle": n, "pool-MCD": n}[variant]
    Ls, Li, Bi, Ni, Ti, Pe = [], [], [], [], [], []
    for _ in range(reps):
        U, F = draw(rng, (n_screen, m), p, chol, eps, 6.0)
        Xs, _ = draw(rng, (n,), p, chol, eps, 6.0)       # SRS of n, the yardstick
        Ls.append(np.sum(Xs.mean(0) ** 2))

        if variant == "oracle":
            mu0, S0inv = mu, Sinv; Us, Fs = U, F
        elif variant == "pool-MCD":
            mu0, S0inv = mcd_pilot(U.reshape(-1, p)); Us, Fs = U, F
        else:                                    # both split variants
            mu0, S0inv = mcd_pilot(U[:h].reshape(-1, p))
            Us, Fs = U[h:], F[h:]                # disjoint from the pilot sets
        Pe.append(np.linalg.norm(mu0 - mu))
        Xi, Fi = pick_inner(Us, Fs, mu0, S0inv)
        mi = Xi.mean(0)
        Li.append(np.sum(mi ** 2)); Bi.append(mi)
        Ni.append(int(Fi.sum())); Ti.append(np.trace(np.cov(Xi, rowvar=False)))

    Ls = np.array(Ls); Li = np.array(Li); Bi = np.array(Bi); B = len(Ls)
    re_relse = np.sqrt(Ls.var(ddof=1) / (B * Ls.mean() ** 2) +
                       Li.var(ddof=1) / (B * Li.mean() ** 2))
    return {"RE": Ls.mean() / Li.mean(), "re_relse": re_relse,
            "pilot_rms": float(np.sqrt(np.mean(np.square(Pe)))),
            "bias": np.linalg.norm(Bi.mean(0)), "rmse": np.sqrt(Li.mean()),
            "N": np.mean(Ni), "trC": np.mean(Ti) * fac,
            "n_used": Xi.shape[0], "n_screen": n_screen * m}


if __name__ == "__main__":
    t0 = time.time(); p = 4
    print("p=4, Sigma=AR(1,0.5), tr(Sigma)=4.  RE is against an SRS of n=50 units")
    print("in every row, so the rows are directly comparable.")
    print("'split-MCD-equal' holds the PILOT SAMPLE SIZE fixed at that of")
    print("'split-MCD' and raises the measured size from n/2 to n.\n")
    for eps in (0.00, 0.20):
        print(f"================= eps = {eps:.2f} =================")
        print(f"{'variant':>16} {'m':>3} {'screened':>9} {'measured':>9} "
              f"{'RE':>7} {'rel.se':>7} {'rms pilot':>10} {'bias':>7} {'RMSE':>7}")
        for m in (5, 10):
            n = max(1, 50 // m) * m
            for k, v in enumerate(("oracle", "pool-MCD",
                                   "split-MCD", "split-MCD-equal")):
                r = run(p, m, n, 1200, eps, v,
                        seed=30000 + 1000 * m + 100 * int(round(100 * eps)) + k)
                print(f"{v:>16} {m:>3} {r['n_screen']:>9} {r['n_used']:>9} "
                      f"{r['RE']:7.3f} {r['re_relse']:7.3f} "
                      f"{r['pilot_rms']:10.4f} {r['bias']:7.4f} {r['rmse']:7.4f}")
        print()

    print("Interpretation: compare 'split-MCD' with 'split-MCD-equal'.  Their")
    print("pilots are fitted on the same number of units, so the difference is")
    print("purely the measured size.  The ratio of their REs is the factor the")
    print("text previously asserted to be 2.")
    print(f"\nelapsed {time.time()-t0:.0f}s")
