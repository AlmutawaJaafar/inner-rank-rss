"""
exp17_poolpilot_theory.py -- does the finite-sample theory survive a REALISTIC pilot?

Run:  python3.10 -u exp17_poolpilot_theory.py | tee exp17.txt   (about 12 minutes)

THE OBJECTION THIS ANSWERS
--------------------------
Theorems 4 and 6 assume an oracle pilot, or at least a pilot independent of the
measured sets.  Every configuration in which the design performs well uses a
MAD pool pilot fitted on all N = nm screened units, which is independent of
neither.  The theory therefore motivates a design whose recommended form it
does not formally cover.  That is the single most common objection to the
paper, and it is answered not by weakening the claims but by measuring how far
the conclusions actually drift.

Four quantities, each with an exact oracle prediction:

  (a) Cov(muhat)  against  (c_1/p) Sigma / n                    [Theorem 4]
      reported as the ratio of traces, and as the invariant discrepancy
      || A^{-1/2} Cov_obs A^{-1/2} - I ||_F with A the predicted covariance.

  (b) pi_1, the probability the measured unit is contaminated, against eps^m
      under near-perfect separation (shift 12) and at the moderate shift 6
      used elsewhere in the paper.                               [Theorem 6]

  (c) E[Sigmahat] against Sigma, where Sigmahat = (p/c_1) S.     [Theorem 4]
      reported as the invariant bias || Sigma^{-1/2} E[Sigmahat] Sigma^{-1/2}
      - I ||_F and the trace ratio.

  (d) coverage of the nominal 95% ellipsoidal region
      n (muhat-mu)' [(c_1/p) Sigmahat]^{-1} (muhat-mu) <= chi2_{p,0.95},
      which is the interval a practitioner would actually build from the
      closed form.  Reported for the oracle pilot and the MAD pool pilot.

A fifth block repeats the contamination comparison at p = 10, since every
contamination experiment in the paper is at p = 4 and the gain scales as
m^{2/p}, so the design should be markedly weaker in higher dimension.
"""
import numpy as np, time, warnings
from scipy import stats, integrate
warnings.filterwarnings("ignore")

SEED = 20260929


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def c1(p, m):
    f = lambda x: stats.chi2.pdf(x, p); F = lambda x: stats.chi2.cdf(x, p)
    hi = stats.chi2.ppf(1 - 1e-13, p)
    return integrate.quad(lambda x: x * m * (1 - F(x)) ** (m - 1) * f(x),
                          0, hi, limit=500)[0]


def sqrtm_inv(A):
    w, V = np.linalg.eigh(A)
    return V @ np.diag(w ** -0.5) @ V.T


def draw(rng, shape, p, chol, eps, shift):
    n = int(np.prod(shape))
    X = rng.standard_normal((n, p)) @ chol.T
    f = rng.random(n) < eps
    k = int(f.sum())
    if k:
        d = np.zeros(p); d[0] = 1.0
        X[f] = shift * d + 0.5 * rng.standard_normal((k, p))
    return X.reshape(*shape, p), f.reshape(shape)


def mad_pilot(X):
    mu0 = np.median(X, axis=0)
    s = np.maximum(1.4826 * np.median(np.abs(X - mu0), axis=0), 1e-8)
    return mu0, np.diag(1.0 / s ** 2)


def pick_inner(U, F, mu0, S0inv):
    D = U - mu0
    d = np.einsum("...i,ij,...j->...", D, S0inv, D)
    o = np.argmin(d, axis=1)
    i = np.arange(U.shape[0])
    return U[i, o], F[i, o]


def run(p, m, n, reps, eps, pilot, shift=6.0, seed=0):
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)
    mu = np.zeros(p)
    k1 = c1(p, m); fac = p / k1
    A = (k1 / p) * Sigma / n                     # Theorem 4 prediction for Cov(muhat)
    Ainv = np.linalg.inv(A)
    Sm12 = sqrtm_inv(Sigma)
    crit = stats.chi2.ppf(0.95, p)

    mus, Sighats, nc, cov_hit, cov_hit_oracleSig = [], [], [], [], []
    for _ in range(reps):
        U, F = draw(rng, (n, m), p, chol, eps, shift)
        if pilot == "oracle":
            mu0, S0inv = mu, Sinv
        else:                                     # MAD pool pilot on all n*m units
            mu0, S0inv = mad_pilot(U.reshape(-1, p))
        Xi, Fi = pick_inner(U, F, mu0, S0inv)
        mh = Xi.mean(0)
        S = np.cov(Xi, rowvar=False)
        Sh = fac * S
        mus.append(mh); Sighats.append(Sh); nc.append(int(Fi.sum()))
        # (d) coverage with the plug-in Sigmahat, and with the oracle Sigma
        try:
            q = n * (mh - mu) @ np.linalg.solve((k1 / p) * Sh, (mh - mu))
            cov_hit.append(q <= crit)
        except np.linalg.LinAlgError:
            cov_hit.append(False)
        cov_hit_oracleSig.append((mh - mu) @ Ainv @ (mh - mu) <= crit)

    mus = np.array(mus); Sighats = np.array(Sighats)
    Cobs = np.cov(mus, rowvar=False)
    Am12 = sqrtm_inv(A)
    ESig = Sighats.mean(0)
    B = reps
    return {
        "trace_ratio_cov": float(np.trace(Cobs) / np.trace(A)),
        "cov_discrep": float(np.linalg.norm(Am12 @ Cobs @ Am12 - np.eye(p), "fro")),
        "bias_norm": float(np.linalg.norm(mus.mean(0))),
        "bias_over_se": float(np.linalg.norm(mus.mean(0)) /
                              np.sqrt(np.trace(A) / B)),
        "sig_discrep": float(np.linalg.norm(Sm12 @ ESig @ Sm12 - np.eye(p), "fro")),
        "sig_trace_ratio": float(np.trace(ESig) / np.trace(Sigma)),
        "pi1": float(np.mean(nc) / n),
        "cover_plugin": float(np.mean(cov_hit)),
        "cover_oracleSig": float(np.mean(cov_hit_oracleSig)),
        "cover_se": float(np.sqrt(0.95 * 0.05 / B)),
    }


if __name__ == "__main__":
    t0 = time.time()
    n, reps = 50, 4000

    print("=" * 78)
    print("A/C/D  Clean data: do Theorem 4's moments and the closed-form")
    print("       interval survive the MAD pool pilot?   p=4, n=50, 4000 reps")
    print("=" * 78)
    print(f"{'m':>3}{'pilot':>12}{'tr ratio':>10}{'cov disc':>10}"
          f"{'||bias||':>10}{'b/se':>7}{'Sig disc':>10}{'Sig tr':>8}"
          f"{'cover':>8}{'cov(or)':>9}")
    for m in (5, 10):
        for j, pl in enumerate(("oracle", "MAD pool")):
            r = run(4, m, n, reps, 0.0, pl, seed=700 + 10 * m + j)
            print(f"{m:>3}{pl:>12}{r['trace_ratio_cov']:>10.3f}"
                  f"{r['cov_discrep']:>10.3f}{r['bias_norm']:>10.4f}"
                  f"{r['bias_over_se']:>7.1f}{r['sig_discrep']:>10.3f}"
                  f"{r['sig_trace_ratio']:>8.3f}"
                  f"{r['cover_plugin']:>8.3f}{r['cover_oracleSig']:>9.3f}")
    print(f"\n  nominal coverage 0.95, Monte Carlo s.e. "
          f"{np.sqrt(0.95*0.05/reps):.4f}")
    print("  'tr ratio'  = tr Cov_obs / tr[(c_1/p)Sigma/n]       (1 = Theorem 4 holds)")
    print("  'cov disc'  = ||A^-1/2 Cov_obs A^-1/2 - I||_F        (0 = exact)")
    print("  'Sig disc'  = ||Sigma^-1/2 E[Sigmahat] Sigma^-1/2 - I||_F")

    print("\n" + "=" * 78)
    print("B  Contamination exclusion under the MAD pool pilot:  pi_1 vs eps^m")
    print("=" * 78)
    print(f"{'m':>3}{'eps':>6}{'shift':>7}{'pilot':>12}{'pi_1 obs':>11}"
          f"{'eps^m':>11}{'ratio':>8}")
    for shift in (12.0, 6.0):
        for m in (5, 10):
            for eps in (0.10, 0.20):
                for j, pl in enumerate(("oracle", "MAD pool")):
                    r = run(4, m, n, 4000, eps, pl, shift=shift,
                            seed=int(800 + 100 * m + 10 * eps * 100 + j + shift))
                    tgt = eps ** m
                    rat = r["pi1"] / tgt if tgt > 0 else float("nan")
                    print(f"{m:>3}{eps:>6.2f}{shift:>7.0f}{pl:>12}"
                          f"{r['pi1']:>11.6f}{tgt:>11.6f}{rat:>8.2f}")
        print()

    print("=" * 78)
    print("E  Dimension: the same contamination comparison at p = 10")
    print("=" * 78)
    print(f"{'p':>3}{'m':>3}{'eps':>6}{'pilot':>12}{'tr ratio':>10}"
          f"{'||bias||':>10}{'pi_1':>10}{'eps^m':>10}")
    for p in (4, 10):
        for m in (5, 10):
            for eps in (0.0, 0.20):
                r = run(p, m, n, 2000, eps, "MAD pool",
                        seed=int(900 + 100 * p + 10 * m + eps * 100))
                print(f"{p:>3}{m:>3}{eps:>6.2f}{'MAD pool':>12}"
                      f"{r['trace_ratio_cov']:>10.3f}{r['bias_norm']:>10.4f}"
                      f"{r['pi1']:>10.6f}{eps**m:>10.6f}")
    print("\n  p=10 uses the same diagonal MAD pilot, which ignores the AR(1)")
    print("  correlations; the gain scales as m^{2/p}, so the design is")
    print("  expected to be markedly weaker here than at p=4.")
    print(f"\nelapsed {time.time()-t0:.0f}s")
