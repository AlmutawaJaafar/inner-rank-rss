"""
exp01_prop1.py  --  Resolves review issue #1.

PROBLEM.  Table 1 of the manuscript reports RE(mu) = 0.9928 at p = 10 with a
seed-level standard error of 0.0020, apparently 3.6 SE below the value 1 that
Proposition 1 asserts exactly.  Either the proposition is wrong, the code is
wrong, or the reported uncertainty is wrong.

DIAGNOSIS.  The RE ratio is a very low-power way to test Proposition 1.  The
proposition's content is the statement

        mu_(r) := E[X | radial rank r] = mu   for every r,

which can be tested directly, and with far more power, by estimating each
mu_(r) from many independent sets: every set of size m contributes one
observation to EVERY rank, so N sets give N observations per rank instead of
one RE value per replication.

This script therefore runs two tests.

  TEST A (mechanism, high power).  Estimate mu_(r) for r = 1..m from N sets and
         test H0: mu_(r) = 0 by Hotelling's T^2, per rank and jointly.
         Also estimate the correction term of the RSS variance identity,
             Delta_r := (1/m) sum_r ||mu_(r)||^2,
         which Proposition 1 says is exactly zero and which is the entire
         source of any RE different from 1.

  TEST B (RE, correct uncertainty).  Estimate RE with a delta-method standard
         error.  The correct analytic SE of the ratio of two independent mean
         losses is
             SE(RE) ~= RE * sqrt( s_A^2/(R*Abar^2) + s_B^2/(R*Bbar^2) ),
         which for ||muhat||^2 with muhat ~ N(0, Sigma/n) equals
             RE * sqrt(2) * sqrt(2 tr(Sigma^2)) / (tr(Sigma) sqrt(R)).
         The manuscript's seed-level SE understates this; the script reports
         both so the discrepancy can be documented rather than hidden.

Run time: about 3 minutes.
"""
import numpy as np
from scipy import stats
import json, time

SEED = 20260918
OUT = "exp01_prop1_results.json"


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def radial_rank_draw(rng, n_sets, m, p, chol, Sinv, block=20000):
    """
    Draw n_sets independent sets of m units and return, for each rank r,
    the selected observations.  Output shape (m, n_sets, p): out[r] holds the
    n_sets observations that were ranked r-th (0-indexed, increasing radius).
    Every set contributes to every rank, which is what gives this test its
    power relative to an RE-based test.
    """
    out = np.empty((m, n_sets, p))
    done = 0
    while done < n_sets:
        b = min(block, n_sets - done)
        U = rng.standard_normal((b, m, p)) @ chol.T
        d = np.einsum("...i,ij,...j->...", U, Sinv, U)       # squared radius
        order = np.argsort(d, axis=1)                         # (b, m)
        # gather U[i, order[i, r], :] for every r
        sel = np.take_along_axis(U, order[:, :, None], axis=1)  # (b, m, p)
        out[:, done:done + b, :] = np.transpose(sel, (1, 0, 2))
        done += b
    return out


def hotelling(X):
    """One-sample Hotelling T^2 test of H0: mean = 0.  Returns (T2, F, df, p)."""
    N, p = X.shape
    xbar = X.mean(axis=0)
    S = np.cov(X, rowvar=False)
    T2 = N * xbar @ np.linalg.solve(S, xbar)
    F = (N - p) / (p * (N - 1)) * T2
    pval = stats.f.sf(F, p, N - p)
    return float(T2), float(F), (p, N - p), float(pval)


def test_A(p, m, n_sets, seed):
    rng = np.random.default_rng([SEED, seed, p])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    sel = radial_rank_draw(rng, n_sets, m, p, chol, Sinv)
    res = {"p": p, "m": m, "n_sets": n_sets, "ranks": []}
    norms = []
    for r in range(m):
        Xr = sel[r]
        T2, F, df, pv = hotelling(Xr)
        mu_r = Xr.mean(axis=0)
        se_r = Xr.std(axis=0, ddof=1) / np.sqrt(n_sets)
        norms.append(float(np.sum(mu_r ** 2)))
        res["ranks"].append({
            "r": r + 1,
            "mu_hat": [round(float(v), 5) for v in mu_r],
            "max_abs_t": round(float(np.max(np.abs(mu_r / se_r))), 3),
            "hotelling_F": round(F, 3), "p_value": round(pv, 4),
            "mean_radius": round(float(np.mean(
                np.einsum("...i,ij,...j->...", Xr, Sinv, Xr))), 4),
        })
    # the correction term of the variance identity
    delta = float(np.mean(norms))
    # reference scale: what a correction term of this size implies for RE
    implied_re = 1.0 / (1.0 - delta / (np.trace(Sigma) / m)) if delta > 0 else 1.0
    res["Delta_hat"] = delta
    res["Delta_null_scale"] = float(np.trace(Sigma) / n_sets)  # E[Delta] under H0:
    # each rank mean averages ONE unit per set, so it has n_sets terms, not m*n_sets.
    # E||muhat_(r)||^2 = tr Cov(X|rank r)/n_sets, and averaging over r gives
    # tr Sigma / n_sets by Proposition 2.
    res["implied_RE_from_Delta"] = implied_re
    # Under H0, n_sets*||mu_r||^2_Sigma^{-1} ~ chi2_p, so m*n_sets*Delta_Sigma ~ chi2_{mp}
    return res


def test_B(p, m, k, reps, seed, block=2000):
    """Relative efficiency of the RSS mean with a delta-method standard error."""
    rng = np.random.default_rng([SEED, 1000 + seed, p])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    n = k * m
    ranks_r = np.tile(np.arange(m), k)
    A, B = [], []                                    # SRS loss, RSS loss
    done = 0
    while done < reps:
        b = min(block, reps - done)
        # ---- RSS arm
        U = rng.standard_normal((b, n, m, p)) @ chol.T
        d = np.einsum("...i,ij,...j->...", U, Sinv, U)
        order = np.argsort(d, axis=2)
        pick = order[:, np.arange(n), ranks_r]                   # (b, n)
        Xr = np.take_along_axis(U, pick[:, :, None, None], axis=2)[:, :, 0, :]
        B.append(np.sum(Xr.mean(axis=1) ** 2, axis=1))
        # ---- SRS arm
        Xs = rng.standard_normal((b, n, p)) @ chol.T
        A.append(np.sum(Xs.mean(axis=1) ** 2, axis=1))
        done += b
    A = np.concatenate(A); B = np.concatenate(B)
    RE = A.mean() / B.mean()
    # delta method on the log ratio of two independent means
    se_log = np.sqrt(A.var(ddof=1) / (len(A) * A.mean() ** 2)
                     + B.var(ddof=1) / (len(B) * B.mean() ** 2))
    lo, hi = RE * np.exp(-1.96 * se_log), RE * np.exp(1.96 * se_log)
    theo = np.trace(Sigma) / n
    return {"p": p, "m": m, "n": n, "reps": reps,
            "RE": float(RE), "se": float(RE * se_log),
            "ci95": [float(lo), float(hi)],
            "mean_loss_SRS": float(A.mean()), "mean_loss_RSS": float(B.mean()),
            "theoretical_mean_loss": float(theo),
            "z_vs_one": float((RE - 1) / (RE * se_log))}


if __name__ == "__main__":
    t0 = time.time(); out = {"test_A": [], "test_B": []}

    print("TEST A  --  direct test of  E[X | rank r] = mu   (Proposition 1)")
    print("          500,000 sets per configuration\n")
    for p in (4, 10):
        r = test_A(p, m=5, n_sets=500_000, seed=p)
        out["test_A"].append(r)
        print(f"  p={p}, m=5, {r['n_sets']:,} sets")
        for row in r["ranks"]:
            print(f"    rank {row['r']}: max|t|={row['max_abs_t']:5.2f}  "
                  f"Hotelling F={row['hotelling_F']:6.3f}  p={row['p_value']:.3f}  "
                  f"E[d^2|rank]={row['mean_radius']:.3f}")
        print(f"    Delta_hat = (1/m)*sum_r ||mu_r||^2 = {r['Delta_hat']:.3e} "
              f"(expected under H0: {r['Delta_null_scale']:.3e})")
        print(f"    implied RE from Delta_hat = {r['implied_RE_from_Delta']:.5f}\n")

    print("TEST B  --  relative efficiency with a delta-method SE")
    print("          50,000 replications per configuration\n")
    for p in (4, 10):
        r = test_B(p, m=5, k=10, reps=50_000, seed=p)
        out["test_B"].append(r)
        print(f"  p={p:2d}  RE = {r['RE']:.4f}  (SE {r['se']:.4f})  "
              f"95% CI [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]  "
              f"z vs 1 = {r['z_vs_one']:+.2f}")
        print(f"        mean loss  SRS {r['mean_loss_SRS']:.5f}   "
              f"RSS {r['mean_loss_RSS']:.5f}   theory {r['theoretical_mean_loss']:.5f}")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwritten to {OUT};  elapsed {time.time()-t0:.0f}s")
