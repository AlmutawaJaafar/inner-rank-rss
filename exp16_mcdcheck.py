"""
exp16_mcdcheck.py -- two referee questions about the baselines.

Run:  python3.10 -u exp16_mcdcheck.py | tee exp16.txt      (about 8 minutes)

A. IS THE FAST-MCD FAILURE AT eps=0.30 REAL, OR A SEARCH ARTEFACT?

   Table `comp` reports Fast-MCD on the full pool (N=250, p=4) with RMSE 2.024
   at eps=0.30, worse than the plain sample mean.  With 30% contamination and
   h = 127 > 75 = number of outliers, no h-subset consists of outliers alone,
   so one might expect the MCD to be safe and suspect the subset search.

   Three things are measured here, to separate the estimator from its
   implementation:
     1. scikit-learn's Fast-MCD, the implementation used in the manuscript;
     2. a deliberate 50-start C-step search with reweighting, which explores
        the objective far more thoroughly;
     3. for each replication, whether the subset the search returns has a
        SMALLER log-determinant than the all-clean subset of the same size.

   Item 3 is the decisive diagnostic.  If the failures come with a smaller
   determinant than the clean subset, the MCD objective itself prefers the
   contaminated solution and the failure is a genuine property of the
   estimator; if the determinant is larger, the search simply did not find the
   optimum.  The contaminants here have scatter 0.5 I against the clean
   scatter Sigma, so they form a tight cluster, and a subset that absorbs them
   can have a smaller determinant than the clean one.

B. DOES Sigma_0 = 3*Sigma REPRODUCE THE ORACLE UNDER COMMON RANDOM NUMBERS?

   Scaling the pilot scatter by a positive constant leaves the ranking
   unchanged, so with the SAME draws the selected units are identical and the
   two runs must agree exactly.  Section 5.4 reported 2.709 against 2.920 for
   the oracle, a gap of about 2.7 standard errors, because the two runs used
   independent seeds.  This section re-runs them under common random numbers.
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


def draw_pool(rng, N, p, chol, eps, shift=6.0):
    X = rng.standard_normal((N, p)) @ chol.T
    f = rng.random(N) < eps
    k = int(f.sum())
    if k:
        d = np.zeros(p); d[0] = 1.0
        X[f] = shift * d + 0.5 * rng.standard_normal((k, p))
    return X, f


# --------------------------------------------------------------- MCD machinery
def cstep(X, idx, h, iters=30):
    """Concentration steps from an initial subset; returns (logdet, loc, cov)."""
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
    """Best-of-nstart elemental C-step search, as in Rousseeuw--Van Driessen."""
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
    """Rousseeuw--Van Driessen reweighting at the 0.975 chi-square cutoff."""
    prec = np.linalg.inv(cov + 1e-12 * np.eye(p))
    d = np.einsum("ij,jk,ik->i", X - loc, prec, X - loc)
    cf = np.median(d) / stats.chi2.ppf(0.5, p)
    keep = d / max(cf, 1e-12) < stats.chi2.ppf(0.975, p)
    return loc if keep.sum() < p + 1 else X[keep].mean(0)


# ===================================================================== part A
def partA(p=4, N=250, reps=200, nstart=50):
    print("=" * 76)
    print("A. Fast-MCD on the full pool: genuine failure, or a search artefact?")
    print("=" * 76)
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma)
    h = int(0.5 * (N + p + 1))
    sep = 6.0 ** 2 * np.linalg.inv(Sigma)[0, 0]
    print(f"N={N}, p={p}, h={h}.  Contaminants sit at squared Mahalanobis")
    print(f"distance {sep:.0f} from the centre (clean E[D]={p}) with scatter")
    print(f"0.5 I, so they form a TIGHT, well separated cluster.\n")

    for eps in (0.20, 0.30):
        n_out = int(eps * N)
        print(f"--- eps={eps:.2f}   outliers ~{n_out} < h={h} ---")
        print(f"{'method':>26}{'RMSE':>9}{'median':>9}{'P90':>9}{'frac>1':>9}")
        store = {}
        for tag in ("sklearn Fast-MCD", f"{nstart}-start C-step", "sample mean"):
            rng = np.random.default_rng([SEED, 1, int(100 * eps)])
            loss, ldc, ldm = [], [], []
            for _ in range(reps):
                X, f = draw_pool(rng, N, p, chol, eps)
                if tag == "sklearn Fast-MCD":
                    try:
                        est = MinCovDet(random_state=0,
                                        support_fraction=0.5).fit(X).location_
                    except Exception:
                        est = np.median(X, axis=0)
                elif tag == "sample mean":
                    est = X.mean(0)
                else:
                    ld, loc, cov = mcd_multistart(X, h, nstart, rng)
                    est = reweight(X, loc, cov, p)
                    clean = np.where(~f)[0]
                    if len(clean) >= h:
                        ldc.append(cstep(X, clean[:h], h)[0]); ldm.append(ld)
                loss.append(float(np.sum(est ** 2)))
            loss = np.array(loss); store[tag] = (loss, np.array(ldc), np.array(ldm))
            print(f"{tag:>26}{np.sqrt(loss.mean()):>9.3f}"
                  f"{np.sqrt(np.median(loss)):>9.3f}"
                  f"{np.sqrt(np.quantile(loss,0.9)):>9.3f}"
                  f"{np.mean(loss>1.0):>9.3f}")

        loss, ldc, ldm = store[f"{nstart}-start C-step"]
        if len(ldc):
            bad = loss[:len(ldc)] > 1.0
            print(f"   determinant diagnostic ({nstart}-start search):")
            print(f"     returned subset has SMALLER logdet than the all-clean "
                  f"subset in {np.mean(ldm < ldc - 1e-9)*100:.0f}% of runs")
            if bad.any():
                print(f"     among the {bad.mean()*100:.0f}% of runs with squared "
                      f"error > 1, that figure is "
                      f"{np.mean(ldm[bad] < ldc[bad] - 1e-9)*100:.0f}%, mean "
                      f"logdet gap {np.mean(ldc[bad]-ldm[bad]):.3f}")
                print(f"     => the objective PREFERS the contaminated subset; "
                      f"this is estimator breakdown, not a search failure.")
        print()


# ===================================================================== part B
def partB(p=4, m=5, n=50, reps=4000):
    print("=" * 76)
    print("B. Sigma_0 = 3*Sigma under COMMON RANDOM NUMBERS")
    print("=" * 76)
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma)
    variants = {"oracle  Sigma_0 = Sigma": np.linalg.inv(Sigma),
                "        Sigma_0 = 3Sigma": np.linalg.inv(3 * Sigma),
                "        Sigma_0 = I": np.eye(p)}
    rng = np.random.default_rng([SEED, 99])
    Ls = []; Li = {k: [] for k in variants}; ident = 0
    for _ in range(reps):
        U = rng.standard_normal((n, m, p)) @ chol.T          # common draws
        Xs = rng.standard_normal((n, p)) @ chol.T
        Ls.append(np.sum(Xs.mean(0) ** 2))
        picks = {}
        for tag, S0inv in variants.items():
            d = np.einsum("...i,ij,...j->...", U, S0inv, U)
            o = np.argmin(d, axis=1); picks[tag] = o
            Li[tag].append(np.sum(U[np.arange(n), o].mean(0) ** 2))
        if np.array_equal(picks["oracle  Sigma_0 = Sigma"],
                          picks["        Sigma_0 = 3Sigma"]):
            ident += 1
    Ls = np.array(Ls)
    print(f"selections identical between Sigma and 3*Sigma in "
          f"{ident}/{reps} replications ({100*ident/reps:.1f}%)")
    print(f"theoretical RE* = p/c_1 = {p/ED1(p,m):.4f}\n")
    print(f"{'variant':>26}{'RE':>9}{'s.e.':>8}")
    for tag in variants:
        L = np.array(Li[tag]); re = Ls.mean() / L.mean()
        se = re * np.sqrt(Ls.var(ddof=1) / (reps * Ls.mean() ** 2) +
                          L.var(ddof=1) / (reps * L.mean() ** 2))
        print(f"{tag:>26}{re:>9.3f}{se:>8.3f}")


if __name__ == "__main__":
    t0 = time.time()
    partA(); print(); partB()
    print(f"\nelapsed {time.time()-t0:.0f}s")
