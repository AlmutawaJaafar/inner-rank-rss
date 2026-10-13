"""exp13b_checks.py -- two small checks for the revision.
 (a) Section 5.5: is the innermost-rank gap at (p,m)=(4,5) the known-mean vs
     estimated-mean difference?  Same design, known-mean S_0, 4000 reps.
 (b) Table 7: linearity of kappa_m(t) in t at a second configuration (p,m)=(10,5).
Run:  python3.10 -u exp13b_checks.py | tee exp13b.txt   (about 1 minute)"""
import numpy as np, sys
sys.argv = [sys.argv[0]]
from exp11_scatter_alloc import order_moments
SEED = 20261002
p, m, n, reps = 4, 5, 60, 4000
c, d = order_moments(p, m); a = c[0] / p
for tag, known in (("estimated mean (as in the paper)", False), ("known mean S_0", True)):
    rng = np.random.default_rng([SEED, int(known)])
    eq, es = [], []
    for _ in range(reps):
        U = rng.standard_normal((n, m, p))
        X = U[np.arange(n), np.argmin((U ** 2).sum(2), 1)]
        Xs = rng.standard_normal((n, p))
        if known:
            Sq, Ss = X.T @ X / n, Xs.T @ Xs / n
        else:
            Sq, Ss = np.cov(X, rowvar=False), np.cov(Xs, rowvar=False)
        eq.append(np.sum((Sq / a - np.eye(p)) ** 2)); es.append(np.sum((Ss - np.eye(p)) ** 2))
    eq, es = np.array(eq), np.array(es); re = es.mean() / eq.mean()
    se = re * np.sqrt(es.var(ddof=1)/(reps*es.mean()**2) + eq.var(ddof=1)/(reps*eq.mean()**2))
    print(f"(a) innermost, (4,5), {tag:>33}: RE {re:.3f} (s.e. {se:.3f});  theory 1.105")
p, m = 10, 5
for t in (0.05, 0.1, 0.2, 0.4):
    rng = np.random.default_rng([SEED, 7, int(1000 * t)])
    beta = np.zeros(p); beta[0] = t; tot = 0.0; tot2 = 0.0; B = 300000; done = 0
    while done < B:
        nb = min(20000, B - done); Z = rng.standard_normal((nb, m, p))
        sel = Z[np.arange(nb), np.argmin(((Z - beta) ** 2).sum(2), 1)][:, 0]
        tot += sel.sum(); tot2 += (sel ** 2).sum(); done += nb
    mu = tot / B; se = np.sqrt(tot2 / B - mu ** 2) / np.sqrt(B)
    print(f"(b) (p,m)=(10,5)  t={t:4.2f}: kappa = {mu/t:.4f} (s.e. {se/t:.4f})")
