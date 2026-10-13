"""
exp08_theory_check.py -- numerical verification of the Week-2 theorems.

T4   Exact scatter consistency.  The selected units are i.i.d. with covariance
     (c_1/p) Sigma, c_1 = E[D_(1:m)], so S is exactly unbiased for (c_1/p)Sigma
     and  Sigmahat = (p/c_1) S  satisfies  E[Sigmahat] = Sigma  for every n >= 2.
     Also checks the closed forms
         p = 2:  RE* = p/c_1 = m     exactly
         large m: RE* ~ p m^{2/p} / (2 Gamma(1+2/p) Gamma(p/2+1)^{2/p}).

T6   Contamination exclusion.  Under perfect radial separation the measured
     unit is contaminated with probability exactly eps^m, so the expected
     number of contaminated units in the sample is n eps^m rather than n eps.
     In general p_1 = m eps INT H(x)^{m-1} dG(x) <= m eps [1-(1-eps)delta]^{m-1}.

BIAS Bias-transmission law.  With pilot mu0 = mu + b, the selected unit is the
     nearest of m spherical points to beta = Sigma^{-1/2} b, with density
         f_sel(z) = m f(z) S_beta(||z-beta||)^{m-1},
     so E[muhat] - mu = c(m,p)*b + o(b).  Checks c(1)=0, monotonicity in m,
     and c -> 1 as m grows.
"""
import numpy as np
from scipy import stats, integrate, special
import time

SEED = 20260922


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def c1(p, m):
    """E[D_(1:m)] for D ~ chi^2_p, by quadrature."""
    f = lambda x: stats.chi2.pdf(x, p); F = lambda x: stats.chi2.cdf(x, p)
    hi = stats.chi2.ppf(1 - 1e-13, p)
    return integrate.quad(lambda x: x * m * (1 - F(x)) ** (m - 1) * f(x),
                          0, hi, limit=500)[0]


def c1_asym(p, m):
    """Large-m approximation of E[D_(1:m)]."""
    return special.gamma(1 + 2 / p) * 2 * special.gamma(p / 2 + 1) ** (2 / p) * m ** (-2 / p)


# ------------------------------------------------------------------ T4
def check_T4(p, m, n, reps, seed=0):
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    fac = p / c1(p, m)
    acc = np.zeros((p, p)); accmu = np.zeros(p)
    for _ in range(reps):
        U = rng.standard_normal((n, m, p)) @ chol.T
        d = np.einsum("...i,ij,...j->...", U, Sinv, U)
        X = U[np.arange(n), np.argmin(d, axis=1)]
        acc += np.cov(X, rowvar=False) * fac
        accmu += X.mean(axis=0)
    Sig_hat = acc / reps
    err = np.abs(Sig_hat - Sigma).max()
    return {"fac": fac, "max_abs_err": err, "rel_err": err / np.abs(Sigma).max(),
            "bias_mu": np.linalg.norm(accmu / reps)}


# ------------------------------------------------------------------ T6
def check_T6(p, m, eps, shift, reps, seed=0):
    rng = np.random.default_rng([SEED, 500 + seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    hit = 0
    for _ in range(reps):
        U = rng.standard_normal((m, p)) @ chol.T
        f = rng.random(m) < eps
        k = int(f.sum())
        if k:
            dvec = np.zeros(p); dvec[0] = 1.0
            U[f] = shift * dvec + 0.5 * rng.standard_normal((k, p))
        d = np.einsum("...i,ij,...j->...", U, Sinv, U)
        hit += int(f[np.argmin(d)])
    return hit / reps


# ------------------------------------------------------------- bias law
def check_bias(p, m, b, reps, seed=0):
    """Mean of the nearest point to beta = b*e1 among m spherical draws."""
    rng = np.random.default_rng([SEED, 900 + seed])
    beta = np.zeros(p); beta[0] = b
    tot = np.zeros(p)
    block = 20000
    done = 0
    while done < reps:
        nb = min(block, reps - done)
        Z = rng.standard_normal((nb, m, p))
        d = ((Z - beta) ** 2).sum(axis=2)
        tot += Z[np.arange(nb), np.argmin(d, axis=1)].sum(axis=0)
        done += nb
    return (tot / reps)[0] / b if b > 0 else 0.0


if __name__ == "__main__":
    t0 = time.time()

    print("=" * 72)
    print("T4  exact scatter consistency:  E[(p/c_1) S] = Sigma")
    print("=" * 72)
    print(f"{'p':>3}{'m':>4}{'n':>5}{'reps':>7}{'p/c_1':>9}{'max|err|':>10}{'rel err':>9}{'bias mu':>9}")
    for (p, m, n, reps) in ((4, 5, 50, 4000), (4, 10, 50, 4000),
                            (10, 5, 50, 4000), (2, 5, 50, 4000)):
        r = check_T4(p, m, n, reps, seed=p * 100 + m)
        print(f"{p:>3}{m:>4}{n:>5}{reps:>7}{r['fac']:>9.4f}"
              f"{r['max_abs_err']:>10.4f}{r['rel_err']:>9.4f}{r['bias_mu']:>9.5f}")

    print("\nclosed forms for RE* = p/c_1")
    print(f"{'p':>3}{'m':>4}{'exact':>10}{'asymptotic':>12}{'p=2 claim m':>13}")
    for p in (2, 4, 10):
        for m in (2, 5, 10, 20, 50):
            ex = p / c1(p, m); asy = p / c1_asym(p, m)
            claim = f"{m}" if p == 2 else "-"
            print(f"{p:>3}{m:>4}{ex:>10.4f}{asy:>12.4f}{claim:>13}")

    print("\n" + "=" * 72)
    print("T6  contamination exclusion:  p_1 = eps^m under perfect separation")
    print("=" * 72)
    print(f"{'m':>3}{'eps':>6}{'shift':>7}{'observed p_1':>14}{'eps^m':>12}"
          f"{'bound m*eps^m':>15}")
    for m in (3, 5, 10):
        for eps in (0.10, 0.20, 0.30):
            obs = check_T6(4, m, eps, 12.0, 400000, seed=m * 10 + int(100 * eps))
            print(f"{m:>3}{eps:>6.2f}{12.0:>7.1f}{obs:>14.6f}"
                  f"{eps**m:>12.6f}{m*eps**m:>15.6f}")
    print("\n  moderate separation (shift = 4), where the bound is not tight:")
    for m in (5,):
        for eps in (0.10, 0.20, 0.30):
            obs = check_T6(4, m, eps, 4.0, 400000, seed=777 + int(100 * eps))
            print(f"{m:>3}{eps:>6.2f}{4.0:>7.1f}{obs:>14.6f}{eps**m:>12.6f}"
                  f"{m*eps**m:>15.6f}")

    print("\n" + "=" * 72)
    print("BIAS law:  E[muhat] - mu = c(m,p) b + o(b);  c(1)=0, c increasing, c->1")
    print("=" * 72)
    print(f"{'p':>3}" + "".join(f"{'m='+str(m):>9}" for m in (1, 2, 5, 10, 20, 50)))
    for p in (2, 4, 10):
        row = f"{p:>3}"
        for m in (1, 2, 5, 10, 20, 50):
            row += f"{check_bias(p, m, 0.05, 400000, seed=p*50+m):>9.4f}"
        print(row)
    print("\n  linearity check at p=4, m=5 (c should not depend on b):")
    for b in (0.02, 0.05, 0.1, 0.2, 0.4):
        print(f"    b={b:5.2f}   c = {check_bias(4, 5, b, 300000, seed=int(1000*b)):.4f}")
    print(f"\nelapsed {time.time()-t0:.0f}s")
