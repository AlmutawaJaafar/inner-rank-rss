"""
exp03_mechanism.py  --  Resolves review issue #5.

PROBLEM.  Proposition 4 assumes that failure of the MCD occurs essentially when
the realised contamination count N exceeds a critical value j*, so that
P(fail) ~= P(N > j*).  The manuscript never tests this.  It is the load-bearing
assumption of the paper's main result and it has two parts, only one of which
is obvious:

  (i)  P(fail | N = j) rises sharply from 0 to 1 around some j*;
  (ii) that conditional curve is THE SAME for RSS and for SRS.

Part (ii) is the non-trivial one.  If RSS changed the conditional failure
probability given the count -- for instance because its selected outliers are
systematically more extreme -- then the whole mechanism would be misattributed,
and the count-variance inequality of Proposition 3 would not explain anything.

WHAT THIS DOES.
  1. Records (N, loss) for both designs across a grid of contamination levels.
  2. Estimates P(fail | N = j) separately for each design, with binomial CIs,
     and tests equality of the two curves (pooled two-proportion z tests).
  3. Estimates j* as the count at which the pooled curve crosses 1/2.
  4. Performs the decomposition that settles the attribution:

         P_fail^pred(RSS) = sum_j P(N_RSS = j) * P(fail | N = j, SRS)

     i.e. it plugs the RSS count distribution into the SRS conditional curve.
     If the predicted value matches the OBSERVED RSS failure probability, then
     the entire RSS effect is attributable to the count distribution -- exactly
     the claim of Propositions 3 and 4 -- and nothing else.

Results are appended to CSV after every cell, so a long run is never lost.

Run time: about 25 minutes.
"""
import numpy as np, csv, os, time, warnings
from sklearn.covariance import MinCovDet
warnings.filterwarnings("ignore")

SEED = 20260918
TAU = 1.0                      # failure threshold on ||muhat - mu||^2
RAW = "exp03_raw.csv"
SUMM = "exp03_summary.csv"


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


def loss(X):
    return float(np.sum(MinCovDet(random_state=0).fit(X).location_ ** 2))


def wilson(x, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = x / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_cell(p, m, k, eps, reps, shift=6.0, seed=0):
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    n = k * m
    rows = []
    for _ in range(reps):
        Xr, Fr = rss_flag(rng, k, m, p, chol, Sinv, eps, shift)
        Xs, Fs = draw_flag(rng, (n,), p, chol, eps, shift)
        rows.append(("RSS", eps, n, m, int(Fr.sum()), loss(Xr)))
        rows.append(("SRS", eps, n, m, int(Fs.sum()), loss(Xs)))
    return rows


def analyse(rows, eps, n, m):
    arr = {d: np.array([(r[4], r[5]) for r in rows if r[0] == d]) for d in ("SRS", "RSS")}
    out = {"eps": eps, "n": n, "m": m}
    for d in ("SRS", "RSS"):
        N, L = arr[d][:, 0].astype(int), arr[d][:, 1]
        out[f"P_{d}"] = float(np.mean(L > TAU))
        out[f"meanN_{d}"] = float(N.mean())
        out[f"sdN_{d}"] = float(N.std(ddof=1))
        out[f"RE_{d}"] = None
    out["RE"] = float(arr["SRS"][:, 1].mean() / arr["RSS"][:, 1].mean())
    out["varratio"] = float(arr["SRS"][:, 0].var(ddof=1) / arr["RSS"][:, 0].var(ddof=1))

    # conditional failure curves
    allN = np.concatenate([arr["SRS"][:, 0], arr["RSS"][:, 0]]).astype(int)
    js = np.arange(allN.min(), allN.max() + 1)
    cond = []
    for j in js:
        row = {"j": int(j)}
        for d in ("SRS", "RSS"):
            N, L = arr[d][:, 0].astype(int), arr[d][:, 1]
            sel = N == j
            cnt = int(sel.sum()); fail = int((L[sel] > TAU).sum())
            row[f"n_{d}"] = cnt
            row[f"p_{d}"] = fail / cnt if cnt else np.nan
            row[f"ci_{d}"] = wilson(fail, cnt)
        # pooled two-proportion z test of equality of the conditional curves
        a, na = row["p_SRS"], row["n_SRS"]
        b, nb = row["p_RSS"], row["n_RSS"]
        if na >= 10 and nb >= 10:
            pp = (a * na + b * nb) / (na + nb)
            se = np.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
            row["z"] = float((a - b) / se) if se > 0 else 0.0
        else:
            row["z"] = np.nan
        cond.append(row)
    out["cond"] = cond

    # j*: smallest j at which the pooled conditional curve exceeds 1/2
    jstar = None
    for row in cond:
        tot = row["n_SRS"] + row["n_RSS"]
        if tot >= 20:
            pooled = (row["p_SRS"] * row["n_SRS"] + row["p_RSS"] * row["n_RSS"]) / tot
            if pooled > 0.5 and jstar is None:
                jstar = row["j"]
    out["jstar"] = jstar

    # ---- attribution: RSS count distribution x SRS conditional curve
    condmap = {row["j"]: row["p_SRS"] for row in cond if row["n_SRS"] >= 10}
    Nr = arr["RSS"][:, 0].astype(int)
    covered = np.array([j in condmap for j in Nr])
    pred = np.mean([condmap[j] for j in Nr[covered]]) if covered.any() else np.nan
    out["P_RSS_predicted"] = float(pred)
    out["coverage"] = float(covered.mean())
    return out


if __name__ == "__main__":
    t0 = time.time()
    grid = [(0.14, 8000), (0.18, 8000), (0.22, 8000), (0.26, 8000), (0.30, 8000)]
    p, m, k = 4, 5, 10
    if not os.path.exists(RAW):
        with open(RAW, "w", newline="") as f:
            csv.writer(f).writerow(["design", "eps", "n", "m", "N_contam", "loss"])
    summ = []
    for i, (eps, reps) in enumerate(grid):
        rows = run_cell(p, m, k, eps, reps, seed=i)
        with open(RAW, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        s = analyse(rows, eps, k * m, m)
        summ.append(s)
        print(f"\n=== eps={eps:.2f}  n={k*m}  m={m}  reps={reps} ===", flush=True)
        print(f"  P_fail  SRS {s['P_SRS']:.4f}   RSS {s['P_RSS']:.4f}   "
              f"RE {s['RE']:.3f}   varratio {s['varratio']:.2f}")
        print(f"  count   SRS mean {s['meanN_SRS']:.2f} sd {s['sdN_SRS']:.2f} | "
              f"RSS mean {s['meanN_RSS']:.2f} sd {s['sdN_RSS']:.2f}")
        print(f"  j* (pooled curve crosses 1/2) = {s['jstar']}")
        print(f"  ATTRIBUTION: predicted P_RSS from SRS conditional curve = "
              f"{s['P_RSS_predicted']:.4f}  vs observed {s['P_RSS']:.4f}  "
              f"(coverage {s['coverage']:.3f})")
        print("   j   n_SRS  P(fail|N=j)_SRS       n_RSS  P(fail|N=j)_RSS     z")
        for row in s["cond"]:
            if row["n_SRS"] >= 25 or row["n_RSS"] >= 25:
                zs = "  n/a" if np.isnan(row["z"]) else f"{row['z']:+5.2f}"
                print(f"  {row['j']:3d}  {row['n_SRS']:6d}  {row['p_SRS']:.3f} "
                      f"[{row['ci_SRS'][0]:.3f},{row['ci_SRS'][1]:.3f}]  "
                      f"{row['n_RSS']:6d}  {row['p_RSS']:.3f} "
                      f"[{row['ci_RSS'][0]:.3f},{row['ci_RSS'][1]:.3f}]  {zs}",
                      flush=True)
    with open(SUMM, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps", "P_SRS", "P_RSS", "P_RSS_predicted", "RE", "varratio",
                    "meanN_SRS", "sdN_SRS", "meanN_RSS", "sdN_RSS", "jstar"])
        for s in summ:
            w.writerow([s["eps"], s["P_SRS"], s["P_RSS"], s["P_RSS_predicted"],
                        s["RE"], s["varratio"], s["meanN_SRS"], s["sdN_SRS"],
                        s["meanN_RSS"], s["sdN_RSS"], s["jstar"]])
    print(f"\nelapsed {time.time()-t0:.0f}s  ->  {RAW}, {SUMM}")
