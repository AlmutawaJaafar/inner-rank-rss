"""
exp02_prop2.py  --  Resolves review issue #4 (and supplies the SEs missing
from the scatter table).

PROBLEM.  Proposition 2 states that radial ranking is *perfect* ranking for the
squared Mahalanobis radius D = (X-mu)' Sigma^{-1} (X-mu), so that the RSS mean
of D attains the classical perfect-ranking efficiency

    RE(D) = [ 1 - (1/(m Var D)) sum_r (E[D_(r)] - E[D])^2 ]^{-1},

with D ~ chi^2_p under normality.  The manuscript never tests this: it reports
relative efficiencies for tr(Sigma_hat) and log det(Sigma_hat), which are
different (and harder) targets.  A proposition that is never checked is an
invitation to a referee.

WHAT THIS DOES.
  (a) Computes E[D_(r)], r = 1..m, by numerical integration against the exact
      order-statistic density of a chi^2_p variate, and from these the
      theoretical RE of Proposition 2.  No simulation error.
  (b) Estimates the same quantity by simulation and compares.  Agreement
      validates Proposition 2; disagreement falsifies it.
  (c) Reports the Takahasi-Wakimoto ceiling (m+1)/2 alongside, since
      Proposition 2 claims RE(D) lies in (1, (m+1)/2].
  (d) Re-runs the scatter functionals (Frobenius, trace, log det) WITH
      delta-method standard errors, which the manuscript's table lacks.

Run time: about 2 minutes.
"""
import numpy as np
from scipy import stats, integrate
import json, time

SEED = 20260918


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


# ---------------------------------------------------------------- (a) theory
def order_stat_moments(p, m):
    """E[D_(r)] for D ~ chi^2_p, r = 1..m, by numerical integration."""
    from math import lgamma, exp
    f = lambda x: stats.chi2.pdf(x, p)
    F = lambda x: stats.chi2.cdf(x, p)
    hi = stats.chi2.ppf(1 - 1e-12, p)
    mus = []
    for r in range(1, m + 1):
        logc = lgamma(m + 1) - lgamma(r) - lgamma(m - r + 1)
        c = exp(logc)
        g = lambda x, r=r, c=c: x * c * F(x) ** (r - 1) * (1 - F(x)) ** (m - r) * f(x)
        val, err = integrate.quad(g, 0, hi, limit=400)
        mus.append(val)
    return np.array(mus)


def prop2_theory(p, m):
    mu_r = order_stat_moments(p, m)
    ED, VarD = p, 2 * p
    corr = np.mean((mu_r - ED) ** 2)          # (1/m) sum_r (mu_(r) - E D)^2
    RE = 1.0 / (1.0 - corr / VarD)
    return {"p": p, "m": m, "E_D_order": [round(float(v), 4) for v in mu_r],
            "sum_check_mean": float(mu_r.mean()), "E_D": ED, "Var_D": VarD,
            "correction": float(corr), "RE_theory": float(RE),
            "TW_ceiling": (m + 1) / 2}


# ----------------------------------------------------------- (b),(d) simulation
def simulate(p, m, k, reps, seed, block=1000):
    rng = np.random.default_rng([SEED, seed, p])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    n = k * m
    ranks_r = np.tile(np.arange(m), k)
    keys = ["D", "fro", "trace", "logdet"]
    A = {key: [] for key in keys}      # SRS
    B = {key: [] for key in keys}      # RSS
    trS, ldS = np.trace(Sigma), np.linalg.slogdet(Sigma)[1]

    def losses(X, store):
        d = np.einsum("...i,ij,...j->...", X, Sinv, X)          # (b, n)
        store["D"].append((d.mean(axis=1) - p) ** 2)
        Xc = X - X.mean(axis=1, keepdims=True)
        S = np.einsum("bni,bnj->bij", Xc, Xc) / (X.shape[1] - 1)
        store["fro"].append(((S - Sigma) ** 2).sum(axis=(1, 2)))
        store["trace"].append((np.trace(S, axis1=1, axis2=2) - trS) ** 2)
        store["logdet"].append((np.linalg.slogdet(S)[1] - ldS) ** 2)

    done = 0
    while done < reps:
        b = min(block, reps - done)
        U = rng.standard_normal((b, n, m, p)) @ chol.T
        d = np.einsum("...i,ij,...j->...", U, Sinv, U)
        order = np.argsort(d, axis=2)
        pick = order[:, np.arange(n), ranks_r]
        Xr = np.take_along_axis(U, pick[:, :, None, None], axis=2)[:, :, 0, :]
        Xs = rng.standard_normal((b, n, p)) @ chol.T
        losses(Xr, B); losses(Xs, A)
        done += b

    out = {}
    for key in keys:
        a = np.concatenate(A[key]); bb = np.concatenate(B[key])
        RE = a.mean() / bb.mean()
        se_log = np.sqrt(a.var(ddof=1) / (len(a) * a.mean() ** 2)
                         + bb.var(ddof=1) / (len(bb) * bb.mean() ** 2))
        out[key] = {"RE": float(RE), "se": float(RE * se_log),
                    "ci95": [float(RE * np.exp(-1.96 * se_log)),
                             float(RE * np.exp(1.96 * se_log))]}
    return out


if __name__ == "__main__":
    t0 = time.time(); res = {"theory": [], "sim": []}
    print("(a) Proposition 2: theoretical RE for the squared radius D ~ chi^2_p\n")
    print(f"{'p':>3} {'m':>3} {'RE_theory':>10} {'(m+1)/2':>9}   E[D_(r)]")
    for p in (4, 10):
        for m in (3, 5, 10):
            th = prop2_theory(p, m); res["theory"].append(th)
            print(f"{p:3d} {m:3d} {th['RE_theory']:10.4f} {th['TW_ceiling']:9.1f}   "
                  + " ".join(f"{v:.2f}" for v in th["E_D_order"]))
    print("\n  (sanity: mean of E[D_(r)] over r must equal E[D] = p)")
    for th in res["theory"]:
        print(f"    p={th['p']:2d} m={th['m']:2d}: mean {th['sum_check_mean']:.6f}"
              f"  vs  p = {th['E_D']}")

    print("\n(b),(d) simulation, m=5, n=50, 20,000 replications, with 95% CIs\n")
    print(f"{'p':>3} {'target':>8} {'RE':>8} {'SE':>7}  {'95% CI':>18}  {'theory':>8}")
    for p in (4, 10):
        s = simulate(p, m=5, k=10, reps=20_000, seed=p)
        res["sim"].append({"p": p, **s})
        th = [t for t in res["theory"] if t["p"] == p and t["m"] == 5][0]
        for key, lab in (("D", "D (P2)"), ("fro", "Frob"),
                         ("trace", "trace"), ("logdet", "logdet")):
            v = s[key]
            ref = f"{th['RE_theory']:8.4f}" if key == "D" else "       -"
            print(f"{p:3d} {lab:>8} {v['RE']:8.4f} {v['se']:7.4f}  "
                  f"[{v['ci95'][0]:6.4f}, {v['ci95'][1]:6.4f}]  {ref}")

    json.dump(res, open("exp02_prop2_results.json", "w"), indent=1)
    print(f"\nelapsed {time.time()-t0:.0f}s")
