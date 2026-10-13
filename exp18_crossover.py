"""
exp18_crossover.py -- where does the inner-rank design actually overtake a
                      PROPERLY SEARCHED minimum covariance determinant?

Run:  python3.10 -u exp18_crossover.py | tee exp18.txt     (about 15 minutes)

WHY THIS EXISTS
---------------
Table `comp` compares the inner-rank mean against scikit-learn's Fast-MCD on
the full pool and finds the ordering reversing at about eps = 0.20.  Section
5.8 then shows that the default search is unreliable: at eps = 0.30 a
deliberate 50-start C-step search cuts the MCD's RMSE from about 2.0 to 1.31.
If the same is true at eps = 0.20, the reported crossover is an artefact of the
baseline's search rather than a property of the estimators, and the headline
claim moves.

Everything here runs under COMMON RANDOM NUMBERS: for each replication one pool
of N = nm units is drawn, and all four estimators see exactly that pool.  The
comparison is therefore paired and the crossover is located without any
between-run noise.

  inner-rank (n)     MAD pool pilot, one unit measured per set: n of N units
  mean (N)           the plain sample mean of the pool
  sklearn MCD (N)    scikit-learn Fast-MCD, the manuscript's baseline
  50-start MCD (N)   best-of-50 elemental C-step search with reweighting

Note the comparison is deliberately unfair to the inner-rank estimator: it uses
n = 50 measured units against the MCD's N = 250.  That is the Model B setting,
where all N units are already measured and only the estimator's cost differs.
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
    f = lambda x: stats.chi2.pdf(x, p); F = lambda x: stats.chi2.cdf(x, p)
    hi = stats.chi2.ppf(1 - 1e-12, p)
    return integrate.quad(lambda x: x * m * (1 - F(x)) ** (m - 1) * f(x),
                          0, hi, limit=400)[0]


def mad_pilot(X):
    mu0 = np.median(X, axis=0)
    s = np.maximum(1.4826 * np.median(np.abs(X - mu0), axis=0), 1e-8)
    return mu0, np.diag(1.0 / s ** 2)


def cstep(X, idx, h, iters=30):
    n, p = X.shape
    for _ in range(iters):
        sub = X[idx]; loc = sub.mean(0); cov = np.cov(sub, rowvar=False)
        try:
            prec = np.linalg.inv(cov + 1e-12 * np.eye(p))
        except np.linalg.LinAlgError:
            return np.inf, loc, cov
        d = np.einsum("ij,jk,ik->i", X - loc, prec, X - loc)
        new = np.argsort(d)[:h]
        if np.array_equal(np.sort(new), np.sort(idx)):
            break
        idx = new
    sub = X[idx]; loc = sub.mean(0); cov = np.cov(sub, rowvar=False)
    s, ld = np.linalg.slogdet(cov)
    return (ld if s > 0 else np.inf), loc, cov


def mcd_multistart(X, h, nstart, rng):
    n, p = X.shape
    best = (np.inf, None, None)
    for _ in range(nstart):
        idx = rng.choice(n, p + 1, replace=False)
        sub = X[idx]; loc = sub.mean(0)
        cov = np.cov(sub, rowvar=False) + 1e-6 * np.eye(p)
        d = np.einsum("ij,jk,ik->i", X - loc, np.linalg.inv(cov), X - loc)
        ld, loc, cov = cstep(X, np.argsort(d)[:h], h)
        if ld < best[0]:
            best = (ld, loc, cov)
    return best


def reweight(X, loc, cov, p):
    prec = np.linalg.inv(cov + 1e-12 * np.eye(p))
    d = np.einsum("ij,jk,ik->i", X - loc, prec, X - loc)
    cf = np.median(d) / stats.chi2.ppf(0.5, p)
    keep = d / max(cf, 1e-12) < stats.chi2.ppf(0.975, p)
    return loc if keep.sum() < p + 1 else X[keep].mean(0)


def run(p, m, n, eps, reps, nstart=50, shift=6.0, seed=0):
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma)
    N = n * m; h = int(0.5 * (N + p + 1))
    keys = ("inner-rank (n)", "mean (N)", "sklearn MCD (N)",
            f"{nstart}-start MCD (N)")
    loss = {k: [] for k in keys}
    for _ in range(reps):
        X = rng.standard_normal((N, p)) @ chol.T          # ONE pool, shared
        f = rng.random(N) < eps
        k = int(f.sum())
        if k:
            dv = np.zeros(p); dv[0] = 1.0
            X[f] = shift * dv + 0.5 * rng.standard_normal((k, p))
        U = X.reshape(n, m, p)

        mu0, S0inv = mad_pilot(X)                         # MAD pool pilot
        D = U - mu0
        d = np.einsum("...i,ij,...j->...", D, S0inv, D)
        Xi = U[np.arange(n), np.argmin(d, axis=1)]
        loss["inner-rank (n)"].append(np.sum(Xi.mean(0) ** 2))
        loss["mean (N)"].append(np.sum(X.mean(0) ** 2))
        try:
            est = MinCovDet(random_state=0, support_fraction=0.5).fit(X).location_
        except Exception:
            est = np.median(X, axis=0)
        loss["sklearn MCD (N)"].append(np.sum(est ** 2))
        ld, loc, cov = mcd_multistart(X, h, nstart, rng)
        loss[f"{nstart}-start MCD (N)"].append(np.sum(reweight(X, loc, cov, p) ** 2))
    return {k: np.array(v) for k, v in loss.items()}


if __name__ == "__main__":
    t0 = time.time()
    p, m, n, reps = 4, 5, 50, 400
    grid = (0.00, 0.10, 0.20, 0.25, 0.30)
    print(f"p={p}, m={m}, n={n}, N={n*m}, {reps} replications, common random")
    print(f"numbers across estimators.  Entries are RMSE = sqrt(mean ||est-mu||^2).\n")
    keys = ("inner-rank (n)", "mean (N)", "sklearn MCD (N)", "50-start MCD (N)")
    print(f"{'estimator':>20}" + "".join(f"{'eps='+f'{e:.2f}':>12}" for e in grid))
    res = {}
    for e in grid:
        res[e] = run(p, m, n, e, reps, seed=int(1000 + 100 * e * 100))
    for k in keys:
        print(f"{k:>20}" + "".join(f"{np.sqrt(res[e][k].mean()):>12.3f}"
                                   for e in grid))
    print()
    print(f"{'paired MSE ratio':>20}" +
          "".join(f"{'':>12}" for _ in grid))
    for base in ("sklearn MCD (N)", "50-start MCD (N)"):
        row = f"{'vs '+base.split()[0]:>20}"
        for e in grid:
            r = res[e][base].mean() / res[e]["inner-rank (n)"].mean()
            row += f"{r:>12.2f}"
        print(row)
    print("\n  ratio > 1 means the inner-rank estimator has the smaller MSE.")
    print("  The crossover against the careful search is where its row passes 1.")

    # ---------------------------------------------------------------- bootstrap
    # The MCD loss is bimodal near breakdown, so the ratio of means needs an
    # uncertainty statement.  Replications are independent and every estimator
    # sees the same pool, so we resample REPLICATION INDICES and recompute the
    # ratio, which keeps the pairing intact.
    B = 4000
    brng = np.random.default_rng([SEED, 4242])
    print(f"\npaired bootstrap of the MSE ratio, {B} resamples, percentile 95% CI")
    print(f"{'baseline':>20}{'eps':>7}{'ratio':>9}{'95% CI':>22}{'P(ratio>1)':>12}")
    for base in ("sklearn MCD (N)", "50-start MCD (N)"):
        for e in grid:
            num = res[e][base]; den = res[e]["inner-rank (n)"]
            n_rep = len(num)
            idx = brng.integers(0, n_rep, size=(B, n_rep))
            boot = num[idx].mean(axis=1) / den[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"{base:>20}{e:>7.2f}"
                  f"{num.mean()/den.mean():>9.2f}"
                  f"{'['+format(lo,'.2f')+', '+format(hi,'.2f')+']':>22}"
                  f"{np.mean(boot > 1.0):>12.3f}")
    print("\n  A CI containing 1 means the crossover is not resolved at that eps.")
    print(f"\nelapsed {time.time()-t0:.0f}s")
