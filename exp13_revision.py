"""
exp13_revision.py -- computations required by the fourth referee report.

PART 1  standard errors for existing results (values must reproduce exactly)
  1a  Table 5: Clopper--Pearson intervals from the observed event counts
  1b  Table 5: a rigorous bound on the clean/contaminated overlap at shift 12
  1c  Table 7: kappa_m(0.05) with Monte Carlo standard errors
  1d  Table 10: relative efficiencies with delta-method standard errors
  1e  Section 5.5: scatter simulation check with standard errors

PART 2  new experiments
  2a  scope of contamination exclusion for the INNER-RANK design (replaces the
      balanced-design Table 12): mean shifts 2..10, scatter inflation, cellwise
  2b  the location--scatter rule of Remark 5 under contamination
  2c  scatter-pilot error: mu0 = mu exactly, Sigma0 != Sigma

Run:  python3.10 -u exp13_revision.py | tee exp13.txt     (about 6 minutes)
"""
import numpy as np, time, warnings, sys
from scipy import stats
from sklearn.covariance import MinCovDet
warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0]]
from exp11_scatter_alloc import order_moments, best_pair

SEED = 20261001


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def relse_ratio(a, b):
    """relative s.e. of mean(a)/mean(b), a and b independent."""
    B = len(a)
    return np.sqrt(np.var(a, ddof=1) / (B * np.mean(a) ** 2) +
                   np.var(b, ddof=1) / (B * np.mean(b) ** 2))


# ======================================================================= 1a
def part_1a():
    print("=" * 76); print("1a  Table 5 -- Clopper-Pearson 95% intervals (B = 400,000 sets)")
    print("=" * 76)
    B = 400000
    obs = {(3, .1): .000935, (3, .2): .008032, (3, .3): .026947,
           (5, .1): .000008, (5, .2): .000320, (5, .3): .002338,
           (10, .1): 0.0, (10, .2): 0.0, (10, .3): .000008}
    allin = True
    print(f"{'m':>3}{'eps':>6}{'events':>8}{'observed':>11}{'95% CI':>26}{'eps^m':>12}  inside")
    for (m, e), o in obs.items():
        k = int(round(o * B))
        lo = stats.beta.ppf(0.025, k, B - k + 1) if k > 0 else 0.0
        hi = stats.beta.ppf(0.975, k + 1, B - k)
        th = e ** m
        ok = lo <= th <= hi; allin &= ok
        print(f"{m:>3}{e:>6.1f}{k:>8d}{o:>11.6f}   [{lo:.7f}, {hi:.7f}]{th:>12.7f}  {ok}")
    print("all theoretical values inside their intervals:", allin)


# ======================================================================= 1b
def part_1b(p=4, shift=12.0, t=60.0):
    """P(D_clean > D_contam) <= P(D_clean > t) + P(D_contam < t), with the
    second term bounded by the triangle inequality in the Sigma^{-1/2} norm."""
    S = ar1(p); Sinv = np.linalg.inv(S)
    mc = np.zeros(p); mc[0] = shift
    center = np.sqrt(mc @ Sinv @ mc)                       # ||Sigma^{-1/2} mu_c||
    lam_min = np.linalg.eigvalsh(S).min()
    # D_c < t  =>  ||0.5 Sigma^{-1/2} Z|| > center - sqrt(t)  =>  ||Z||^2 > 4 lam_min (center-sqrt t)^2
    thr = 4 * lam_min * (center - np.sqrt(t)) ** 2
    b1 = stats.chi2.sf(t, p); b2 = stats.chi2.sf(thr, p)
    print("\n" + "=" * 76); print("1b  overlap bound for shift 12 (per clean/contaminated pair)")
    print("=" * 76)
    print(f"  P(D_clean > {t:.0f}) = {b1:.2e};  P(D_contam < {t:.0f}) <= {b2:.2e};  "
          f"pair overlap <= {b1 + b2:.2e};  per set (m=10, <= 25 pairs) <= {25 * (b1 + b2):.2e}")


# ======================================================================= 1c
def part_1c():
    SEED08 = 20260922
    print("\n" + "=" * 76); print("1c  Table 7 -- kappa_m(0.05) with Monte Carlo s.e.")
    print("=" * 76)
    print(f"{'p':>3}" + "".join(f"{'m='+str(m):>17}" for m in (1, 2, 5, 10, 20, 50)))
    for p in (2, 4, 10):
        row = f"{p:>3}"
        for m in (1, 2, 5, 10, 20, 50):
            b = 0.05; reps = 400000; seed = p * 50 + m
            rng = np.random.default_rng([SEED08, 900 + seed])
            beta = np.zeros(p); beta[0] = b
            tot = np.zeros(p); tot2 = 0.0; done = 0
            while done < reps:
                nb = min(20000, reps - done)
                Z = rng.standard_normal((nb, m, p))
                d = ((Z - beta) ** 2).sum(axis=2)
                sel = Z[np.arange(nb), np.argmin(d, axis=1)]
                tot += sel.sum(axis=0); tot2 += np.sum(sel[:, 0] ** 2)
                done += nb
            mean0 = tot[0] / reps
            sd0 = np.sqrt(tot2 / reps - mean0 ** 2)
            row += f"   {mean0/b:+.4f} ({sd0/np.sqrt(reps)/b:.4f})"
        print(row)


# ======================================================================= 1d
def part_1d():
    SEED09 = 20260925
    print("\n" + "=" * 76); print("1d  Table 10 -- RE with delta-method relative s.e.")
    print("=" * 76)
    p, m, n, reps = 4, 5, 50, 6000
    for nu, seed in ((None, 1), (10, 10), (5, 5), (3, 3)):
        rng = np.random.default_rng([SEED09, seed])
        Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)

        def draw(shape):
            Z = rng.standard_normal((*shape, p)) @ chol.T
            if nu is None:
                return Z
            W = rng.chisquare(nu, size=(*shape, 1)) / nu
            return Z / np.sqrt(W)
        Ls, Li = [], []
        for _ in range(reps):
            U = draw((n, m))
            d = np.einsum("...i,ij,...j->...", U, Sinv, U)
            X = U[np.arange(n), np.argmin(d, axis=1)]
            Xs = draw((n,))
            Ls.append(np.sum(Xs.mean(0) ** 2)); Li.append(np.sum(X.mean(0) ** 2))
        Ls, Li = np.array(Ls), np.array(Li)
        re = Ls.mean() / Li.mean()
        tag = "Gaussian" if nu is None else f"t_{nu}"
        note = "  (s.e. not meaningful: loss has infinite variance)" if nu == 3 else ""
        print(f"  {tag:>9}: RE {re:.4f}  s.e. {re*relse_ratio(Ls, Li):.4f}{note}")


# ======================================================================= 1e
def part_1e():
    SEED11 = 20260927
    print("\n" + "=" * 76); print("1e  Section 5.5 -- scatter simulation check with s.e.")
    print("=" * 76)
    for p, m in ((4, 5), (4, 10)):
        c, d = order_moments(p, m)
        _, qopt = best_pair(c, d, p, m)
        for k, (tag, q) in enumerate((("innermost", np.eye(m)[0]),
                                      ("balanced", np.full(m, 1 / m)),
                                      ("optimal", qopt))):
            n, reps = 60, 4000
            rng = np.random.default_rng([SEED11, 1000 * p + 10 * m + k])
            a = q @ c / p
            counts = np.round(q * n).astype(int); counts[-1] = n - counts[:-1].sum()
            ranks = np.repeat(np.arange(m), counts)
            eq, es = [], []
            for _ in range(reps):
                U = rng.standard_normal((n, m, p))
                o = np.argsort((U ** 2).sum(axis=2), axis=1)
                X = U[np.arange(n), o[np.arange(n), ranks]]
                eq.append(np.sum((np.cov(X, rowvar=False) / a - np.eye(p)) ** 2))
                Xs = rng.standard_normal((n, p))
                es.append(np.sum((np.cov(Xs, rowvar=False) - np.eye(p)) ** 2))
            eq, es = np.array(eq), np.array(es)
            re = es.mean() / eq.mean()
            print(f"  p={p} m={m:>2} {tag:>10}: RE {re:.3f}  s.e. {re*relse_ratio(es, eq):.3f}")


# ======================================================================= helpers for part 2
def contam(rng, N, p, chol, eps, kind, par):
    X = rng.standard_normal((N, p)) @ chol.T
    f = rng.random(N) < eps
    idx = np.where(f)[0]
    if len(idx):
        if kind == "shift":
            e1 = np.zeros(p); e1[0] = par
            X[idx] = e1 + 0.5 * rng.standard_normal((len(idx), p))
        elif kind == "scatter":
            X[idx] = np.sqrt(par) * (rng.standard_normal((len(idx), p)) @ chol.T)
        elif kind == "cellwise":
            j = rng.integers(0, p, size=len(idx))
            X[idx, j] += par
    return X, f


def mad_pilot(X):
    mu0 = np.median(X, axis=0)
    s = np.maximum(1.4826 * np.median(np.abs(X - mu0), axis=0), 1e-12)
    return mu0, np.diag(1.0 / s ** 2)


def ranks_by(U, mu0, S0inv):
    D = U - mu0
    return np.argsort(np.einsum("nmi,ij,nmj->nm", D, S0inv, D), axis=1)


# ======================================================================= 2a
def part_2a(p=4, m=5, n=50, eps=0.20, reps=1000):
    print("\n" + "=" * 76)
    print("2a  scope of contamination exclusion for the INNER-RANK design")
    print(f"    p={p}, m={m}, n={n}, eps={eps}, {reps} reps; RE vs SRS mean, RE_MCD vs")
    print("    Fast-MCD location on an independent SRS of n (definitions of Table 4)")
    print("=" * 76)
    S = ar1(p); chol = np.linalg.cholesky(S); Sinv = np.linalg.inv(S)
    cells = [("shift", 2), ("shift", 3), ("shift", 4), ("shift", 6), ("shift", 10),
             ("scatter", 3), ("scatter", 5), ("cellwise", 4)]
    print(f"{'contamination':>14}{'par':>5} | {'oracle RE':>10}{'RE_MCD':>8}{'N':>6} | "
          f"{'MAD-pool RE':>12}{'RE_MCD':>8}{'N':>6}")
    for ci, (kind, par) in enumerate(cells):
        rng = np.random.default_rng([SEED, 1, ci])
        Ls, Lm, Lo, Lp, No, Np = [], [], [], [], [], []
        for _ in range(reps):
            pool, f = contam(rng, n * m, p, chol, eps, kind, par)
            U = pool.reshape(n, m, p); F = f.reshape(n, m)
            Xs, _ = contam(rng, n, p, chol, eps, kind, par)
            Ls.append(np.sum(Xs.mean(0) ** 2))
            try:
                Lm.append(np.sum(MinCovDet(random_state=0).fit(Xs).location_ ** 2))
            except Exception:
                Lm.append(np.sum(np.median(Xs, 0) ** 2))
            o = ranks_by(U, np.zeros(p), Sinv)[:, 0]
            Lo.append(np.sum(U[np.arange(n), o].mean(0) ** 2)); No.append(F[np.arange(n), o].sum())
            mu0, S0 = mad_pilot(pool)
            o = ranks_by(U, mu0, S0)[:, 0]
            Lp.append(np.sum(U[np.arange(n), o].mean(0) ** 2)); Np.append(F[np.arange(n), o].sum())
        Ls, Lm, Lo, Lp = map(np.array, (Ls, Lm, Lo, Lp))
        print(f"{kind:>14}{par:>5} | {Ls.mean()/Lo.mean():>10.3f}{Lm.mean()/Lo.mean():>8.3f}"
              f"{np.mean(No):>6.2f} | {Ls.mean()/Lp.mean():>12.3f}{Lm.mean()/Lp.mean():>8.3f}"
              f"{np.mean(Np):>6.2f}   [rel.se <= {max(relse_ratio(Ls,Lo),relse_ratio(Ls,Lp),relse_ratio(Lm,Lo),relse_ratio(Lm,Lp)):.3f}]")


# ======================================================================= 2b
def part_2b(p=4, m=5, n=50, reps=1000):
    c, d = order_moments(p, m)
    rstar = int(np.argmin(d / c ** 2 - 1))                # 0-based index of r*
    print("\n" + "=" * 76)
    print(f"2b  the location--scatter rule under contamination: rank 1 vs rank r*={rstar+1}")
    print(f"    invariant loss of the scatter estimate, p={p}, m={m}, n={n}, {reps} reps, shift 6")
    print("=" * 76)
    S = ar1(p); chol = np.linalg.cholesky(S); Sinv = np.linalg.inv(S)
    w, V = np.linalg.eigh(S); Sm12 = V @ np.diag(w ** -0.5) @ V.T
    loss = lambda H: np.sum((Sm12 @ H @ Sm12 - np.eye(p)) ** 2)
    print(f"{'eps':>5} | {'rank1 orc':>10}{'r* orc':>9}{'rank1 MAD':>10}{'r* MAD':>9}"
          f" | {'SRS cov':>9}{'MCD cov':>9}")
    for ei, eps in enumerate((0.0, 0.10, 0.20)):
        rng = np.random.default_rng([SEED, 2, ei])
        acc = {k: [] for k in ("1o", "so", "1p", "sp", "srs", "mcd")}
        for _ in range(reps):
            pool, _ = contam(rng, n * m, p, chol, eps, "shift", 6.0)
            U = pool.reshape(n, m, p)
            Xs, _ = contam(rng, n, p, chol, eps, "shift", 6.0)
            for tag, (mu0, S0) in (("o", (np.zeros(p), Sinv)), ("p", mad_pilot(pool))):
                o = ranks_by(U, mu0, S0)
                acc["1" + tag].append(loss(p / c[0] * np.cov(U[np.arange(n), o[:, 0]], rowvar=False)))
                acc["s" + tag].append(loss(p / c[rstar] * np.cov(U[np.arange(n), o[:, rstar]], rowvar=False)))
            acc["srs"].append(loss(np.cov(Xs, rowvar=False)))
            try:
                acc["mcd"].append(loss(MinCovDet(random_state=0).fit(Xs).covariance_))
            except Exception:
                acc["mcd"].append(np.nan)
        r = {k: np.nanmean(v) for k, v in acc.items()}
        se = {k: np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v))) for k, v in acc.items()}
        print(f"{eps:>5.2f} | {r['1o']:>10.3f}{r['so']:>9.3f}{r['1p']:>10.3f}{r['sp']:>9.3f}"
              f" | {r['srs']:>9.3f}{r['mcd']:>9.3f}")
        print("      s.e. " + "  ".join(f"{k}:{se[k]:.3f}" for k in ("1o","so","1p","sp","srs","mcd")))


# ======================================================================= 2c
def part_2c(p=4, m=5, n=50, reps=4000):
    print("\n" + "=" * 76)
    print("2c  scatter-pilot error with the centre known (mu0 = mu), clean data")
    print("=" * 76)
    S = ar1(p); chol = np.linalg.cholesky(S)
    for k, (tag, S0) in enumerate((("Sigma0 = Sigma (oracle)", S), ("Sigma0 = I", np.eye(p)),
                                   ("Sigma0 = AR(0.9)", ar1(p, 0.9)), ("Sigma0 = 3 Sigma", 3 * S))):
        rng = np.random.default_rng([SEED, 3, k])
        S0inv = np.linalg.inv(S0)
        Ls, Li, Bi = [], [], []
        for _ in range(reps):
            U = (rng.standard_normal((n * m, p)) @ chol.T).reshape(n, m, p)
            o = ranks_by(U, np.zeros(p), S0inv)[:, 0]
            X = U[np.arange(n), o]
            Xs = rng.standard_normal((n, p)) @ chol.T
            Ls.append(np.sum(Xs.mean(0) ** 2)); Li.append(np.sum(X.mean(0) ** 2)); Bi.append(X.mean(0))
        Ls, Li = np.array(Ls), np.array(Li)
        re = Ls.mean() / Li.mean()
        print(f"  {tag:>24}: RE {re:.3f} (s.e. {re*relse_ratio(Ls,Li):.3f})   |bias| {np.linalg.norm(np.mean(Bi,0)):.4f}")


if __name__ == "__main__":
    t0 = time.time()
    part_1a(); part_1b(); part_1c(); part_1d(); part_1e()
    part_2a(); part_2b(); part_2c()
    print(f"\nelapsed {time.time()-t0:.0f}s")
