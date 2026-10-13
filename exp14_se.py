"""
exp14_se.py -- Monte Carlo standard errors for Tables 3, 4, 7 and 12, from the
identical random draws that produced them (no new random numbers are consumed).

Run:  python3.10 -u exp14_se.py | tee exp14.txt      (about 6 minutes)
"""
import numpy as np, sys, time
sys.argv = [sys.argv[0]]
import exp05_allocation as e5
import exp09_nonelliptical as e9
import exp13_revision as e13

def rel(a, b):
    B = len(a)
    return np.sqrt(np.var(a, ddof=1)/(B*np.mean(a)**2) + np.var(b, ddof=1)/(B*np.mean(b)**2))

t0 = time.time()
print("=" * 72); print("Table 3 -- clean, oracle ranking, 6000 reps"); print("=" * 72)
for p, m in ((4, 5), (4, 10), (10, 5)):
    r = e5.run(p, m, k=max(1, 50 // m), reps=6000, eps=0.0, seed=p * 10 + m)
    S, I = r["acc"]["SRS"], r["acc"]["INNER"]
    re = S["loss"].mean() / I["loss"].mean()
    fac = p / r["ED1"]
    ct = I["trS"].mean() * fac; ct_se = I["trS"].std(ddof=1) * fac / np.sqrt(len(I["trS"]))
    print(f"  p={p:>2} m={m:>2}: RE {re:.3f} (s.e. {re*rel(S['loss'], I['loss']):.3f});"
          f"  corrected trace {ct:.3f} (s.e. {ct_se:.3f})")

print("\n" + "=" * 72); print("Table 4 -- contamination, oracle ranking, 1500 reps"); print("=" * 72)
for eps in (0.10, 0.20, 0.30):
    r = e5.run(4, 5, k=10, reps=1500, eps=eps, use_mcd=True, seed=int(100 * eps))
    A = r["acc"]
    for key in (("BAL", "INNER") if eps == 0.10 else ("INNER",)):
        re = A["SRS"]["loss"].mean() / A[key]["loss"].mean()
        rm = A["SRS"]["mcd"].mean() / A[key]["mcd"].mean()
        print(f"  eps={eps:.2f} {key:>5}: RE {re:.3f} (s.e. {re*rel(A['SRS']['loss'], A[key]['loss']):.3f});"
              f"  RE(MCD) {rm:.3f} (s.e. {rm*rel(A['SRS']['mcd'], A[key]['mcd']):.3f})")

print("\n" + "=" * 72); print("Table 12 -- skewed populations, 4000 reps"); print("=" * 72)
for alpha in (0.0, 1.0, 2.0, 5.0, 10.0):
    r = e9.run_skewed(4, 5, 50, 4000, alpha, seed=int(10 * alpha))
    print(f"  skew-normal a={alpha:>4}: RE(MSE) {r['RE_mse']:.3f} (s.e. {r['RE_mse']*r['relse']:.3f})")
r = e9.run_skewed(4, 5, 50, 4000, 0.0, seed=99, heavy=True)
print(f"  lognormal         : RE(MSE) {r['RE_mse']:.3f} (s.e. {r['RE_mse']*r['relse']:.3f})")

print("\n" + "=" * 72); print("Table 7 -- scatter rule; means with s.e."); print("=" * 72)
e13.part_2b()
print(f"\nelapsed {time.time()-t0:.0f}s")
