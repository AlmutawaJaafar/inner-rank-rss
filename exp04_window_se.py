"""
exp04_window_se.py  --  Resolves review issues #6, #7 and #8.

PROBLEM.
  #6  The window, set-size and scope tables in the manuscript carry no Monte
      Carlo standard errors, yet claims are made from them ("both set sizes
      peak at eps = 0.18").  Without SEs those claims are unsupported.
  #7  The cell n = 50, eps = 0.26 reports P_SRS = P_RSS = 0.431 exactly, which
      the criterion says should be a strict inequality.  It needs enough
      replications to resolve.
  #8  The MCD cells moved by up to 25% between two seed sets at identical
      settings (RE 1.64 -> 2.09, 1.99 -> 1.69).  1000 replications is not
      enough for a heavy-tailed loss.

WHAT THIS DOES.  Re-runs the three headline experiments at 5000 replications
with bootstrap standard errors and Wilson intervals for the failure
probabilities, writing one CSV row per cell as soon as it completes so that an
interrupted run loses nothing.

    PART 1  window:      m in {5,10} x eps grid            (22 cells)
    PART 2  n-scaling:   n in {50,100,200} x eps grid      (18 cells)
    PART 3  set size:    m in {3,5,7,10} x eps in {.10,.20,.30}  (12 cells)

RUN TIME.  Roughly 6-9 hours in total on a laptop.  Run it overnight, or run
the parts separately:  python exp04_window_se.py 1     (etc.)
Resuming: completed cells are skipped automatically on restart.
"""
import numpy as np, csv, os, sys, time, warnings
from sklearn.covariance import MinCovDet
warnings.filterwarnings("ignore")

SEED = 20260918
TAU = 1.0
REPS = 5000
BOOT = 500
OUT = "exp04_results.csv"
HEADER = ["part", "p", "n", "m", "k", "eps", "reps",
          "RE", "RE_se", "RE_lo", "RE_hi",
          "P_SRS", "P_SRS_lo", "P_SRS_hi", "P_RSS", "P_RSS_lo", "P_RSS_hi",
          "dP", "dP_se", "dP_z",
          "varratio", "varratio_se", "meanN_SRS", "sdN_SRS", "sdN_RSS"]


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def draw_flag(rng, shape, p, chol, eps, shift):
    n = int(np.prod(shape))
    X = rng.standard_normal((n, p)) @ chol.T
    f = rng.random(n) < eps
    k = int(f.sum())
    if k:
        d = np.zeros(p); d[0] = 1.0
        X[f] = shift * d + 0.5 * rng.standard_normal((k, p))
    return X.reshape(*shape, p), f.reshape(shape)


def rss_flag(rng, k, m, p, chol, Sinv, eps, shift):
    ns = k * m
    r = np.tile(np.arange(m), k)
    U, F = draw_flag(rng, (ns, m), p, chol, eps, shift)
    s = np.einsum("...i,ij,...j->...", U, Sinv, U)
    o = np.argsort(s, axis=-1)
    pick = o[np.arange(ns), r]
    return U[np.arange(ns), pick], F[np.arange(ns), pick]


def wilson(x, n, z=1.96):
    ph = x / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cell(part, p, m, k, eps, reps=REPS, shift=6.0, seed=0):
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    n = k * m
    lr = np.empty(reps); ls = np.empty(reps)
    cr = np.empty(reps, int); cs = np.empty(reps, int)
    for b in range(reps):
        Xr, Fr = rss_flag(rng, k, m, p, chol, Sinv, eps, shift)
        Xs, Fs = draw_flag(rng, (n,), p, chol, eps, shift)
        cr[b], cs[b] = Fr.sum(), Fs.sum()
        lr[b] = np.sum(MinCovDet(random_state=0).fit(Xr).location_ ** 2)
        ls[b] = np.sum(MinCovDet(random_state=0).fit(Xs).location_ ** 2)

    RE = ls.mean() / lr.mean()
    br = np.random.default_rng(12345)
    idx = br.integers(0, reps, size=(BOOT, reps))
    bs = ls[idx].mean(1) / lr[idx].mean(1)
    vr = cs.var(ddof=1) / cr.var(ddof=1)
    bv = cs[idx].var(axis=1, ddof=1) / cr[idx].var(axis=1, ddof=1)

    fs, fr = int((ls > TAU).sum()), int((lr > TAU).sum())
    Ps, Pr = fs / reps, fr / reps
    dse = np.sqrt(Ps * (1 - Ps) / reps + Pr * (1 - Pr) / reps)
    return [part, p, n, m, k, eps, reps,
            round(RE, 4), round(bs.std(), 4),
            round(np.quantile(bs, .025), 4), round(np.quantile(bs, .975), 4),
            round(Ps, 4), *[round(v, 4) for v in wilson(fs, reps)],
            round(Pr, 4), *[round(v, 4) for v in wilson(fr, reps)],
            round(Pr - Ps, 4), round(dse, 4),
            round((Pr - Ps) / dse, 2) if dse > 0 else 0.0,
            round(vr, 3), round(bv.std(), 3),
            round(cs.mean(), 2), round(cs.std(ddof=1), 2), round(cr.std(ddof=1), 2)]


def build_grid():
    g, i = [], 0
    epsw = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
    for m in (5, 10):
        for e in epsw:
            g.append((1, 4, m, round(50 / m), e, i)); i += 1
    for ntar in (50, 100, 200):
        for e in (0.10, 0.14, 0.18, 0.22, 0.26, 0.30):
            g.append((2, 4, 5, ntar // 5, e, i)); i += 1
    for e in (0.10, 0.20, 0.30):
        for m in (3, 5, 7, 10):
            g.append((3, 4, m, max(1, round(50 / m)), e, i)); i += 1
    return g


if __name__ == "__main__":
    want = set(int(a) for a in sys.argv[1:]) or {1, 2, 3}
    if not os.path.exists(OUT):
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
    done = set()
    with open(OUT) as f:
        for row in csv.DictReader(f):
            done.add((int(row["part"]), int(row["m"]), int(row["k"]), float(row["eps"])))

    t0 = time.time()
    grid = [g for g in build_grid() if g[0] in want]
    print(f"{len(grid)} cells requested, {len(done)} already complete\n", flush=True)
    print(f"{'part':>4} {'m':>3} {'n':>4} {'eps':>5} | {'RE':>6} {'se':>6} | "
          f"{'P_SRS':>6} {'P_RSS':>6} {'dP':>7} {'z':>6} | {'varrat':>6}", flush=True)
    for part, p, m, k, eps, seed in grid:
        if (part, m, k, eps) in done:
            continue
        row = cell(part, p, m, k, eps, seed=seed)
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"{row[0]:>4} {row[3]:>3} {row[2]:>4} {row[5]:>5.2f} | "
              f"{row[7]:>6.3f} {row[8]:>6.3f} | {row[11]:>6.3f} {row[15]:>6.3f} "
              f"{row[19]:>+7.4f} {row[21]:>+6.2f} | {row[22]:>6.2f}", flush=True)
    print(f"\nelapsed {time.time()-t0:.0f}s  ->  {OUT}")
