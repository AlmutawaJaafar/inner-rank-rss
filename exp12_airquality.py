"""
exp12_airquality.py -- real-data application on the UCI Air Quality dataset.

Data: De Vito et al. (2008), UCI Machine Learning Repository, "Air Quality".
9357 hourly records from a multisensor device in an Italian city, March 2004 to
April 2005.  Cheap metal-oxide sensors (PT08.S*) run alongside certified
reference analysers (the *(GT) columns).  Missing values are coded -200.

PART A  (Model A: expensive measurement, cheap ranking)
  For each pollutant, rank sets of m hours by the cheap SENSOR and measure the
  REFERENCE analyser at the chosen rank (balanced design).  The balanced RSS
  mean is unbiased whatever the skewness, so raw concentrations are used.  The
  observed RE is compared with the normal-theory concomitant prediction
  (Stokes, 1977)
        RE = 1 / (1 - rho^2 T_m),   T_m = (1/m) sum_r E[Z_(r:m)]^2,
  Z standard normal, rho the sensor--analyser correlation.

PART B  (Model B: the inner-rank design on real data)
  The four reference analysers give a fully measured 4-vector.  Concentrations
  are right-skewed, and the paper shows skewness makes the inner-rank design
  inadmissible, so we follow its own advice and work on log-concentrations,
  reporting their marginal skewness.  The finite population is the set of
  complete hours; its mean log-vector is the target.  Contamination, when
  present, is a faulty NOx analyser reading six robust standard deviations high.

Run:  python3.10 -u exp12_airquality.py | tee exp12.txt    (about 3 minutes)
"""
import numpy as np, pandas as pd, time, warnings, sys
from scipy import stats, integrate
from sklearn.covariance import MinCovDet
warnings.filterwarnings("ignore")

SEED = 20260930
PATH = sys.argv[1] if len(sys.argv) > 1 else "AirQualityUCI.csv"


def load(path):
    d = pd.read_csv(path, sep=";", decimal=",")
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")].dropna(how="all")
    return d.replace(-200, np.nan)


def normal_T(m):
    """T_m = (1/m) sum_r E[Z_(r:m)]^2 for standard normal order statistics."""
    from math import lgamma, exp
    tot = 0.0
    for r in range(1, m + 1):
        k = exp(lgamma(m + 1) - lgamma(r) - lgamma(m - r + 1))
        g = lambda z, r=r, k=k: z * k * stats.norm.cdf(z) ** (r - 1) * \
            stats.norm.sf(z) ** (m - r) * stats.norm.pdf(z)
        e = integrate.quad(g, -12, 12, limit=400)[0]
        tot += e * e
    return tot / m


# ---------------------------------------------------------------- PART A
def part_a(d, reps=5000):
    pairs = [("C6H6(GT)", "PT08.S2(NMHC)", "benzene"),
             ("CO(GT)", "PT08.S1(CO)", "CO"),
             ("NOx(GT)", "PT08.S3(NOx)", "NOx"),
             ("NO2(GT)", "PT08.S4(NO2)", "NO2")]
    print("=" * 76)
    print("PART A -- Model A: rank by cheap sensor, measure reference analyser")
    print("         balanced RSS mean vs SRS mean of the same measured size n")
    print("=" * 76)
    print(f"{'pollutant':>9} {'N pop':>6} {'rho':>7} {'m':>3} {'n':>4} "
          f"{'RE obs':>8} {'(se)':>7} {'RE Stokes':>10}")
    rows = []
    for k, (gt, sen, name) in enumerate(pairs):
        pop = d[[gt, sen]].dropna().to_numpy()
        y, x = pop[:, 0], pop[:, 1]
        rho = np.corrcoef(y, x)[0, 1]
        mu = y.mean()
        for m in (3, 5):
            T = normal_T(m)
            for n in (30, 60):
                rng = np.random.default_rng([SEED, k, m, n])
                ranks = np.tile(np.arange(m), n // m)
                idx = rng.integers(0, len(y), size=(reps, n, m))
                o = np.argsort(x[idx], axis=2)            # rank by the sensor
                pick = np.take_along_axis(idx, o[:, np.arange(n), ranks][..., None],
                                          axis=2)[..., 0]
                rss = y[pick].mean(axis=1)
                srs = y[rng.integers(0, len(y), size=(reps, n))].mean(axis=1)
                a, b = (srs - mu) ** 2, (rss - mu) ** 2
                re = a.mean() / b.mean()
                se = re * np.sqrt(a.var(ddof=1) / (reps * a.mean() ** 2) +
                                  b.var(ddof=1) / (reps * b.mean() ** 2))
                pred = 1.0 / (1.0 - rho ** 2 * T)
                rows.append((name, len(y), rho, m, n, re, se, pred))
                print(f"{name:>9} {len(y):>6} {rho:>+7.3f} {m:>3} {n:>4} "
                      f"{re:>8.3f} {se:>7.3f} {pred:>10.3f}")
    return rows


# ---------------------------------------------------------------- PART B
def mad_pilot(X):
    mu0 = np.median(X, axis=0)
    s = np.maximum(1.4826 * np.median(np.abs(X - mu0), axis=0), 1e-12)
    return mu0, np.diag(1.0 / s ** 2)


def part_b(d, reps=300, m=5, n=50):
    gt = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
    L = np.log(d[gt].dropna().to_numpy())
    mu = L.mean(axis=0)
    rs = 1.4826 * np.median(np.abs(L - np.median(L, axis=0)), axis=0)
    print("\n" + "=" * 76)
    print("PART B -- Model B: inner-rank design on the 4-vector of log reference")
    print(f"          concentrations; population = {len(L)} complete hours")
    print("=" * 76)
    print("marginal skewness, raw vs log:")
    raw = d[gt].dropna().to_numpy()
    for j, g in enumerate(gt):
        print(f"   {g:9s}  raw {stats.skew(raw[:, j]):+6.2f}   log {stats.skew(L[:, j]):+6.2f}")
    shift = np.zeros(4); shift[2] = 6 * rs[2]                 # faulty NOx analyser
    N = n * m
    methods = ["mean (n, SRS)", "mean (N)", "median (N)", "MCD (N)",
               "MCD (n, SRS)", "inner-rank (n)"]
    out = {}
    for e_i, eps in enumerate((0.0, 0.10, 0.20)):
        rng = np.random.default_rng([SEED, 99, e_i])
        loss = {k: [] for k in methods}; bias = {k: [] for k in methods}
        secs = {k: 0.0 for k in methods}
        for _ in range(reps):
            pool = L[rng.integers(0, len(L), size=N)].copy()
            f = rng.random(N) < eps
            pool[f] = pool[f] + shift + 0.05 * rng.standard_normal((f.sum(), 4))
            sets = pool.reshape(n, m, 4)
            sub = pool[rng.choice(N, n, replace=False)]
            ests = {}
            t = time.perf_counter(); ests["mean (n, SRS)"] = sub.mean(0)
            secs["mean (n, SRS)"] += time.perf_counter() - t
            t = time.perf_counter(); ests["mean (N)"] = pool.mean(0)
            secs["mean (N)"] += time.perf_counter() - t
            t = time.perf_counter(); ests["median (N)"] = np.median(pool, 0)
            secs["median (N)"] += time.perf_counter() - t
            t = time.perf_counter()
            ests["MCD (N)"] = MinCovDet(random_state=0).fit(pool).location_
            secs["MCD (N)"] += time.perf_counter() - t
            t = time.perf_counter()
            ests["MCD (n, SRS)"] = MinCovDet(random_state=0).fit(sub).location_
            secs["MCD (n, SRS)"] += time.perf_counter() - t
            t = time.perf_counter()
            mu0, S0 = mad_pilot(pool)
            D = sets - mu0
            dd = np.einsum("nmi,ij,nmj->nm", D, S0, D)
            ests["inner-rank (n)"] = sets[np.arange(n), np.argmin(dd, 1)].mean(0)
            secs["inner-rank (n)"] += time.perf_counter() - t
            for k in methods:
                loss[k].append(np.sum((ests[k] - mu) ** 2))
                bias[k].append(ests[k] - mu)
        out[eps] = {k: (np.sqrt(np.mean(loss[k])),
                        np.linalg.norm(np.mean(bias[k], 0)),
                        1000 * secs[k] / reps,
                        np.std(loss[k], ddof=1) / (2 * np.mean(loss[k]) * np.sqrt(reps)))
                    for k in methods}
    base = "MCD (N)"
    for eps, res in out.items():
        print(f"\n  eps = {eps:.2f}   (RMSE of the log-location estimate; "
              f"'rel' relative to {base})")
        print(f"    {'method':>15} {'RMSE':>8} {'rel':>6} {'|bias|':>8} {'ms':>7}")
        for k in methods:
            r, b, ms, rse = res[k]
            print(f"    {k:>15} {r:>8.4f} {r/res[base][0]:>6.2f} {b:>8.4f} {ms:>7.2f}   rel.se {rse:.3f}")
    return out


if __name__ == "__main__":
    t0 = time.time()
    d = load(PATH)
    print(f"loaded {len(d)} hourly records, {d['Date'].iloc[0]} to {d['Date'].iloc[-1]}\n")
    part_a(d)
    part_b(d)
    print(f"\nelapsed {time.time()-t0:.0f}s")
