"""
Feasibility test for a RANK-ALLOCATION theorem.

THE ARGUMENT.
Proposition 1 says E[X | radial rank r] = mu for EVERY r.  Hence the RSS mean
is unbiased for mu under ANY allocation q = (q_1,...,q_m) over ranks, not only
the balanced one.  That is a licence to optimise the allocation.

Under ellipticity, conditioning on the radial rank conditions only on the
radius, and X = mu + R * Sigma^{1/2} U with U uniform on the sphere and
independent of R.  Therefore

    Cov(X | rank r) = (E[D_(r)] / p) * Sigma,      D = (X-mu)'Sigma^{-1}(X-mu),

so for an allocation q the RSS mean has

    Cov(muhat_q) = (1/n) * (sum_r q_r E[D_(r)] / p) * Sigma,

    RE(q) = p / sum_r q_r E[D_(r)],

which is MAXIMISED by putting all mass on rank 1 (the innermost unit), giving

    RE* = p / E[D_(1)].

Predictions to check (E[D_(r)] from the chi^2_p order statistics):
    p=4,  m=5  : E[D_(1)] = 1.4014  ->  RE* = 2.854
    p=4,  m=10 : E[D_(1)] = 0.9310  ->  RE* = 4.296
    p=10, m=5  : E[D_(1)] = 5.4430  ->  RE* = 1.837

Separately, the contamination probability of the selected unit under the
inner-rank design is p_1, which is near zero when outliers are radially
extreme.  So the SAME allocation should minimise variance AND contamination
exposure simultaneously -- no efficiency/robustness trade-off.

This script tests: unbiasedness, the variance formula, the scatter consistency
factor p/E[D_(1)], contamination exposure, and failure probability.
"""
import numpy as np
from scipy import stats, integrate
from sklearn.covariance import MinCovDet
import warnings, time
warnings.filterwarnings("ignore")

SEED = 20260919


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def ED_order(p, m):
    from math import lgamma, exp
    f = lambda x: stats.chi2.pdf(x, p); F = lambda x: stats.chi2.cdf(x, p)
    hi = stats.chi2.ppf(1 - 1e-12, p)
    out = []
    for r in range(1, m + 1):
        c = exp(lgamma(m + 1) - lgamma(r) - lgamma(m - r + 1))
        g = lambda x, r=r, c=c: x * c * F(x)**(r-1) * (1-F(x))**(m-r) * f(x)
        out.append(integrate.quad(g, 0, hi, limit=400)[0])
    return np.array(out)


def draw(rng, shape, p, chol, eps, shift):
    n = int(np.prod(shape))
    X = rng.standard_normal((n, p)) @ chol.T
    f = rng.random(n) < eps
    k = int(f.sum())
    if k:
        d = np.zeros(p); d[0] = 1.0
        X[f] = shift * d + 0.5 * rng.standard_normal((k, p))
    return X.reshape(*shape, p), f.reshape(shape)


def sample_alloc(rng, n, m, p, chol, Sinv, ranks, eps=0.0, shift=6.0):
    """Select, from each of n sets of size m, the unit at the given rank."""
    U, F = draw(rng, (n, m), p, chol, eps, shift)
    d = np.einsum("...i,ij,...j->...", U, Sinv, U)
    o = np.argsort(d, axis=1)
    pick = o[np.arange(n), ranks]
    return U[np.arange(n), pick], F[np.arange(n), pick]


def run(p, m, k, reps, eps=0.0, use_mcd=False, seed=0):
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    n = k * m
    ED = ED_order(p, m)
    bal = np.tile(np.arange(m), k)          # balanced allocation
    inner = np.zeros(n, int)                # all mass on rank 1
    res = {"ED1": ED[0], "RE_pred": p / ED[0]}

    acc = {kk: {"mu": [], "loss": [], "trS": [], "N": [], "mcd": []}
           for kk in ("SRS", "BAL", "INNER")}
    for _ in range(reps):
        Xs, Fs = draw(rng, (n,), p, chol, eps, 6.0)
        Xb, Fb = sample_alloc(rng, n, m, p, chol, Sinv, bal, eps)
        Xi, Fi = sample_alloc(rng, n, m, p, chol, Sinv, inner, eps)
        for key, X, Fl in (("SRS", Xs, Fs), ("BAL", Xb, Fb), ("INNER", Xi, Fi)):
            mu = X.mean(axis=0)
            acc[key]["mu"].append(mu)
            acc[key]["loss"].append(np.sum(mu ** 2))
            acc[key]["trS"].append(np.trace(np.cov(X, rowvar=False)))
            acc[key]["N"].append(int(Fl.sum()))
            if use_mcd:
                acc[key]["mcd"].append(
                    float(np.sum(MinCovDet(random_state=0).fit(X).location_ ** 2)))
    for key in acc:
        a = acc[key]
        a["mu"] = np.array(a["mu"]); a["loss"] = np.array(a["loss"])
        a["trS"] = np.array(a["trS"]); a["N"] = np.array(a["N"])
        a["mcd"] = np.array(a["mcd"]) if use_mcd else None
    res["acc"] = acc
    res["trSigma"] = np.trace(Sigma)
    return res


def summarise(tag, r, use_mcd=False):
    acc = r["acc"]; s = acc["SRS"]["loss"].mean()
    print(f"\n--- {tag} ---")
    print(f"  predicted RE for the inner-rank design = p/E[D_(1)] = {r['RE_pred']:.3f}")
    print(f"  {'design':>6} {'RE(mu)':>8} {'bias norm':>10} {'E tr(S)':>9} "
          f"{'mean N':>7} {'sd N':>6}" + (f" {'RE(MCD)':>8}" if use_mcd else ""))
    for key in ("SRS", "BAL", "INNER"):
        a = acc[key]
        re = s / a["loss"].mean()
        bias = np.linalg.norm(a["mu"].mean(axis=0))
        line = (f"  {key:>6} {re:8.3f} {bias:10.5f} {a['trS'].mean():9.3f} "
                f"{a['N'].mean():7.2f} {a['N'].std(ddof=1):6.2f}")
        if use_mcd:
            line += f" {acc['SRS']['mcd'].mean()/a['mcd'].mean():8.3f}"
        print(line)
    print(f"  true tr(Sigma) = {r['trSigma']:.3f};  "
          f"consistency factor p/E[D_(1)] = {r['RE_pred']:.3f} applied to INNER "
          f"gives {acc['INNER']['trS'].mean()*r['RE_pred']:.3f}")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 74)
    print("CLEAN DATA: does the inner-rank allocation attain RE = p/E[D_(1)]?")
    print("=" * 74)
    for p, m in ((4, 5), (4, 10), (10, 5)):
        r = run(p, m, k=max(1, 50 // m), reps=6000, eps=0.0, seed=p * 10 + m)
        summarise(f"p={p}, m={m}, n={max(1,50//m)*m}, clean, 6000 reps", r)

    print("\n" + "=" * 74)
    print("CONTAMINATED: does the same allocation also minimise exposure?")
    print("=" * 74)
    for eps in (0.10, 0.20, 0.30):
        r = run(4, 5, k=10, reps=1500, eps=eps, use_mcd=True, seed=int(100 * eps))
        summarise(f"p=4, m=5, n=50, eps={eps}, 1500 reps", r, use_mcd=True)

    print(f"\nelapsed {time.time()-t0:.0f}s")
