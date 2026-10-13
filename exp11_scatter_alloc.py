"""
exp11_scatter_alloc.py -- the optimal allocation for the SCATTER matrix.

For an allocation q over radial ranks, with c_r = E[D_(r)], d_r = E[D_(r)^2],

    Sigmahat_q = S / a_q,   a_q = sum_r q_r c_r / p           (unbiased)

    E || Sigmahat_q - Sigma ||_F^2  =  (D_q - C_q / p) / (n a_q^2),
          D_q = sum_r q_r d_r,   C_q = sum_r q_r c_r^2.

The location objective was LINEAR in q, so its optimum was a vertex (the
innermost rank).  This one is a RATIO -- linear over squared-linear -- so a
vertex is no longer guaranteed.  But fixing s = a_q turns the problem into an LP
with two equality constraints (sum q = 1, a_q = s), whose basic solutions have
at most two non-zero entries.  Hence:

    THEOREM.  The Frobenius-optimal allocation for the scatter matrix is
    supported on AT MOST TWO ranks.

This script computes c_r, d_r exactly, searches every vertex and every pair
of ranks (optimising the mixing weight), reports the optimum, and checks it by
simulation.

Run:  python3.10 -u exp11_scatter_alloc.py | tee exp11.txt    (about 2 minutes)
"""
import numpy as np
from scipy import stats, integrate, optimize
from math import lgamma, exp
import itertools, time

SEED = 20260927


def order_moments(p, m):
    """c_r = E[D_(r)], d_r = E[D_(r)^2] for D ~ chi^2_p, by quadrature."""
    f = lambda x: stats.chi2.pdf(x, p); F = lambda x: stats.chi2.cdf(x, p)
    hi = stats.chi2.ppf(1 - 1e-13, p)
    c, d = [], []
    for r in range(1, m + 1):
        k = exp(lgamma(m + 1) - lgamma(r) - lgamma(m - r + 1))
        dens = lambda x, r=r, k=k: k * F(x) ** (r - 1) * (1 - F(x)) ** (m - r) * f(x)
        c.append(integrate.quad(lambda x: x * dens(x), 0, hi, limit=500)[0])
        d.append(integrate.quad(lambda x: x * x * dens(x), 0, hi, limit=500)[0])
    return np.array(c), np.array(d)


def J(q, c, d, p):
    """n * Frobenius MSE of Sigmahat_q (Sigma = I)."""
    a = q @ c / p
    return (q @ d - (q @ c ** 2) / p) / a ** 2


def J_srs(p):
    return p * (p + 1)          # Gaussian, known mean: (tr S^2 + (tr S)^2) at Sigma = I


def best_pair(c, d, p, m):
    best = (np.inf, None)
    for r in range(m):                                   # vertices
        q = np.zeros(m); q[r] = 1.0
        v = J(q, c, d, p)
        if v < best[0]:
            best = (v, q)
    for r, s in itertools.combinations(range(m), 2):     # edges
        def g(lam):
            q = np.zeros(m); q[r] = lam; q[s] = 1 - lam
            return J(q, c, d, p)
        res = optimize.minimize_scalar(g, bounds=(0, 1), method="bounded")
        if res.fun < best[0]:
            q = np.zeros(m); q[r] = res.x; q[s] = 1 - res.x
            best = (res.fun, q)
    return best


def best_general(c, d, p, m, starts=40, seed=0):
    """Unconstrained check over the full simplex, to confirm the <=2 support."""
    rng = np.random.default_rng(seed); best = np.inf
    for _ in range(starts):
        z0 = rng.normal(size=m)
        def obj(z):
            w = np.exp(z - z.max())          # stable softmax
            return J(w / w.sum(), c, d, p)
        res = optimize.minimize(obj, z0, method="Nelder-Mead",
                                options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-12})
        best = min(best, res.fun)
    return best


def simulate(p, m, n, q, reps, seed=0):
    """Empirical Frobenius MSE of Sigmahat_q with estimated mean."""
    rng = np.random.default_rng([SEED, seed])
    c, _ = order_moments(p, m)
    a = q @ c / p
    counts = np.round(q * n).astype(int); counts[-1] = n - counts[:-1].sum()
    ranks = np.repeat(np.arange(m), counts)
    err_q, err_s = [], []
    for _ in range(reps):
        U = rng.standard_normal((n, m, p))
        dd = (U ** 2).sum(axis=2)
        o = np.argsort(dd, axis=1)
        X = U[np.arange(n), o[np.arange(n), ranks]]
        S = np.cov(X, rowvar=False)
        err_q.append(np.sum((S / a - np.eye(p)) ** 2))
        Xs = rng.standard_normal((n, p))
        err_s.append(np.sum((np.cov(Xs, rowvar=False) - np.eye(p)) ** 2))
    return np.mean(err_s) / np.mean(err_q)


if __name__ == "__main__":
    t0 = time.time()
    for p in (2, 4, 10):
        for m in (3, 5, 10):
            c, d = order_moments(p, m)
            cv2 = d / c ** 2 - 1
            vert = [J(np.eye(m)[r], c, d, p) for r in range(m)]
            bal = J(np.full(m, 1 / m), c, d, p)
            val, q = best_pair(c, d, p, m)
            gen = best_general(c, d, p, m, seed=p * 10 + m)
            print(f"\np={p}, m={m}")
            print("  CV^2 of D_(r):        " + " ".join(f"{v:7.3f}" for v in cv2))
            print("  RE(vertex r) vs SRS:  " + " ".join(f"{J_srs(p)/v:7.3f}" for v in vert))
            print(f"  RE(balanced) = {J_srs(p)/bal:.4f}")
            print(f"  RE(optimal)  = {J_srs(p)/val:.4f}   support q = "
                  + " ".join(f"{x:.3f}" for x in q))
            print(f"  general-simplex check: RE = {J_srs(p)/gen:.4f} "
                  f"({'matches' if abs(gen-val)/val < 1e-4 else 'DIFFERS'})")

    print("\n" + "=" * 70)
    print("simulation check, n = 60, 4000 reps (estimated mean)")
    print("=" * 70)
    for p, m in ((4, 5), (4, 10)):
        c, d = order_moments(p, m)
        val, q = best_pair(c, d, p, m)
        for k, (tag, qq) in enumerate((("innermost e_1", np.eye(m)[0]),
                                       ("balanced", np.full(m, 1 / m)),
                                       ("optimal", q))):
            re_th = J_srs(p) / J(qq, c, d, p)
            # deterministic seed: never use hash() of a string, which Python
            # randomises per process and so cannot be reproduced
            re_sim = simulate(p, m, 60, qq, 4000, seed=1000 * p + 10 * m + k)
            print(f"  p={p} m={m:>2}  {tag:>14}:  theory {re_th:6.3f}   "
                  f"simulated {re_sim:6.3f}")
    print(f"\nelapsed {time.time()-t0:.0f}s")
