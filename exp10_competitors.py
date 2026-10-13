"""
exp10_competitors.py -- the comparison that decides the practical claim.

So far the inner-rank design has only been compared against simple random
sampling and against Fast-MCD applied to an SRS sample.  Neither is the
competitor a practitioner would actually use.  Under Model B the whole screened
pool of N = n*m units is already measured, so the honest question is:

    given N fully observed units, how does inner-rank subsampling plus a plain
    sample mean compare with a real robust estimator applied to all N, and at
    what computational cost?

COMPETITORS
  mean (N)         sample mean on the whole pool                [no robustness]
  median (N)       coordinatewise median on the pool            [cheap, robust]
  OGK (N)          orthogonalised Gnanadesikan-Kettenring on the pool
                   (Maronna and Zamar, 2002), implemented here
  MCD (N)          Fast-MCD on the whole pool                   [the benchmark]
  MCD (n, SRS)     Fast-MCD on a random subsample of size n     [cheap-ish]
  INNER (n)        inner-rank subsample of size n, plain sample mean,
                   ranking from a MAD pilot fitted on the pool  [proposed]

The pilot deliberately uses the MAD, not the MCD: an MCD pilot would cost as
much as the estimator it is meant to replace, which would defeat the purpose.
Cost is reported as mean wall-clock seconds per estimate, so the accuracy
columns can be read against the price paid for them.

Run:  python3.10 -u exp10_competitors.py | tee exp10.txt    (about 12 minutes)
"""
import numpy as np, time, warnings
from sklearn.covariance import MinCovDet
warnings.filterwarnings("ignore")

SEED = 20260926


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def draw(rng, shape, p, chol, eps, shift):
    n = int(np.prod(shape))
    X = rng.standard_normal((n, p)) @ chol.T
    f = rng.random(n) < eps
    k = int(f.sum())
    if k:
        d = np.zeros(p); d[0] = 1.0
        X[f] = shift * d + 0.5 * rng.standard_normal((k, p))
    return X.reshape(*shape, p)


# ------------------------------------------------------------------- OGK
def _mad(x, axis=0):
    med = np.median(x, axis=axis)
    return 1.4826 * np.median(np.abs(x - med), axis=axis), med


def ogk(X, reweight=True):
    """
    Orthogonalised Gnanadesikan-Kettenring estimator (Maronna and Zamar, 2002),
    using the MAD as the auxiliary scale and the coordinatewise median as the
    auxiliary location.
    """
    n, p = X.shape
    sd, _ = _mad(X)
    sd = np.maximum(sd, 1e-12)
    Y = X / sd
    U = np.eye(p)
    for j in range(p):
        for k in range(j + 1, p):
            sp, _ = _mad((Y[:, j] + Y[:, k])[:, None]); sp = sp[0]
            sm, _ = _mad((Y[:, j] - Y[:, k])[:, None]); sm = sm[0]
            U[j, k] = U[k, j] = (sp ** 2 - sm ** 2) / 4
    w, E = np.linalg.eigh(U)
    Z = Y @ E
    gam, mz = _mad(Z)
    gam = np.maximum(gam, 1e-12)
    A = (sd[:, None] * E)
    mu = A @ mz
    Sig = A @ np.diag(gam ** 2) @ A.T
    if reweight:                       # one hard-rejection reweighting step
        try:
            d = np.einsum("ni,ij,nj->n", X - mu, np.linalg.inv(Sig), X - mu)
            cut = np.median(d) * stats_chi2_med(p)
            keep = d <= cut
            if keep.sum() > p + 1:
                mu = X[keep].mean(0)
                Sig = np.cov(X[keep], rowvar=False)
        except np.linalg.LinAlgError:
            pass
    return mu, Sig


_CHI2MED = {2: 1.3863, 4: 3.3567, 10: 9.3418}


def stats_chi2_med(p):
    from scipy import stats
    return stats.chi2.ppf(0.975, p) / stats.chi2.ppf(0.5, p)


# ------------------------------------------------------------------- designs
def mad_pilot(X):
    mu0 = np.median(X, axis=0)
    s = np.maximum(1.4826 * np.median(np.abs(X - mu0), axis=0), 1e-12)
    return mu0, np.diag(1.0 / s ** 2)


def inner_rank(pool_sets, mu0, S0inv):
    D = pool_sets - mu0
    d = np.einsum("...i,ij,...j->...", D, S0inv, D)
    o = np.argmin(d, axis=1)
    return pool_sets[np.arange(pool_sets.shape[0]), o]


# ------------------------------------------------------------------- driver
def run(p, m, n, eps, reps, shift=6.0, seed=0):
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma)
    N = n * m
    methods = ["mean (N)", "median (N)", "OGK (N)", "MCD (N)",
               "MCD (n, SRS)", "INNER (n)"]
    loss = {k: [] for k in methods}
    secs = {k: 0.0 for k in methods}

    for _ in range(reps):
        sets = draw(rng, (n, m), p, chol, eps, shift)   # the screened pool
        pool = sets.reshape(-1, p)
        sub = pool[rng.choice(N, n, replace=False)]     # random subsample

        t = time.perf_counter(); mu = pool.mean(0)
        secs["mean (N)"] += time.perf_counter() - t
        loss["mean (N)"].append(np.sum(mu ** 2))

        t = time.perf_counter(); mu = np.median(pool, axis=0)
        secs["median (N)"] += time.perf_counter() - t
        loss["median (N)"].append(np.sum(mu ** 2))

        t = time.perf_counter()
        try:
            mu, _ = ogk(pool)
        except Exception:
            mu = np.median(pool, axis=0)
        secs["OGK (N)"] += time.perf_counter() - t
        loss["OGK (N)"].append(np.sum(mu ** 2))

        t = time.perf_counter()
        try:
            mu = MinCovDet(random_state=0).fit(pool).location_
        except Exception:
            mu = np.median(pool, axis=0)
        secs["MCD (N)"] += time.perf_counter() - t
        loss["MCD (N)"].append(np.sum(mu ** 2))

        t = time.perf_counter()
        try:
            mu = MinCovDet(random_state=0).fit(sub).location_
        except Exception:
            mu = np.median(sub, axis=0)
        secs["MCD (n, SRS)"] += time.perf_counter() - t
        loss["MCD (n, SRS)"].append(np.sum(mu ** 2))

        t = time.perf_counter()
        mu0, S0inv = mad_pilot(pool)
        mu = inner_rank(sets, mu0, S0inv).mean(0)
        secs["INNER (n)"] += time.perf_counter() - t
        loss["INNER (n)"].append(np.sum(mu ** 2))

    out = []
    for k in methods:
        a = np.array(loss[k])
        out.append({"relse": float(a.std(ddof=1) / (2 * a.mean() * np.sqrt(len(a)))),
                    "method": k, "rmse": float(np.sqrt(a.mean())),
                    "ms": 1000 * secs[k] / reps})
    base = [o for o in out if o["method"] == "MCD (N)"][0]["rmse"]
    for o in out:
        o["rel_rmse"] = o["rmse"] / base
    return out


if __name__ == "__main__":
    t0 = time.time()
    for (p, m, n, reps) in ((4, 5, 50, 500), (4, 10, 50, 300)):
        N = n * m
        print("=" * 78)
        print(f"p={p}, m={m}, measured n={n}, screened pool N={N}, {reps} reps")
        print("RMSE of the location estimate; 'rel' is relative to MCD on the "
              "full pool")
        print("=" * 78)
        for eps in (0.0, 0.10, 0.20, 0.30):
            print(f"\n  eps = {eps:.2f}")
            print(f"    {'method':>14}{'RMSE':>9}{'rel':>8}{'ms/est':>9}")
            for o in run(p, m, n, eps, reps, seed=int(100 * eps) + m):
                print(f"    {o['method']:>14}{o['rmse']:>9.4f}"
                      f"{o['rel_rmse']:>8.2f}{o['ms']:>9.2f}   rel.se {o['relse']:.3f}")
    print(f"\nelapsed {time.time()-t0:.0f}s")
