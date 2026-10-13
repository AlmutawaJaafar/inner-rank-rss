"""
make_figures.py -- the three figures of the manuscript.

Figure 1 is computed from scratch (quadrature).  Figures 2 and 3 plot values
from the logged runs; each array is annotated with the script that produced it.

Run:  python3.10 make_figures.py      (writes fig_theory.pdf, fig_competitors.pdf,
                                        fig_airquality.pdf in the current folder)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, integrate, special

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "lines.linewidth": 1.4,
})
BLUE, RED, GREEN, GREY, BLACK = "#1f4e79", "#b03a2e", "#2e7d32", "0.55", "0.1"


def c1(p, m):
    """E[D_(1:m)], D ~ chi^2_p, via the survival integral."""
    S = lambda x: stats.chi2.sf(x, p) ** m
    return integrate.quad(S, 0, np.inf, limit=500)[0]


def c1_asym(p, m):
    return special.gamma(1 + 2 / p) * 2 * special.gamma(p / 2 + 1) ** (2 / p) * m ** (-2 / p)


# ============================================================ Figure 1
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
ms = np.arange(2, 51)
for p, col in ((2, BLUE), (4, RED), (10, GREEN)):
    ex = [p / c1(p, m) for m in ms]
    asy = [p / c1_asym(p, m) for m in ms]
    ax[0].plot(ms, ex, color=col, label=rf"$p={p}$, exact")
    if p != 2:   # for p = 2 the asymptotic form is exact and coincides
        ax[0].plot(ms, asy, color=col, ls="--", lw=1.0, label=rf"$p={p}$, asymptotic")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xticks([2, 5, 10, 20, 50]); ax[0].set_xticklabels(["2", "5", "10", "20", "50"])
ax[0].set_yticks([1, 2, 5, 10, 20, 50]); ax[0].set_yticklabels(["1", "2", "5", "10", "20", "50"])
ax[0].minorticks_off()
ax[0].set_xlabel("set size $m$ (log scale)")
ax[0].set_ylabel(r"$\mathrm{RE}^\ast = p/c_1(m,p)$ (log scale)")
ax[0].set_title("(a) optimal relative efficiency", fontsize=9, loc="left")
ax[0].legend(frameon=False, ncol=1, loc="upper left")

mg = np.linspace(2, 10, 200)
# observed values: code/exp08_theory_check.py, 400,000 sets per cell, shift 12
obs = {0.10: {3: 0.000935, 5: 0.000008},
       0.20: {3: 0.008032, 5: 0.000320},
       0.30: {3: 0.026947, 5: 0.002338, 10: 0.000008}}
for eps, col, mk in ((0.10, BLUE, "o"), (0.20, RED, "s"), (0.30, GREEN, "^")):
    ax[1].plot(mg, eps ** mg, color=col, label=rf"$\varepsilon^m$, $\varepsilon={eps:.1f}$")
    xs, ys = zip(*sorted(obs[eps].items()))
    ax[1].plot(xs, ys, mk, color=col, ms=5, mfc="white", mew=1.2)
ax[1].set_yscale("log")
ax[1].set_xlabel("set size $m$")
ax[1].set_ylabel(r"$\pi_1$ (log scale)")
ax[1].set_title("(b) contamination exclusion", fontsize=9, loc="left")
ax[1].legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("fig_theory.pdf")
plt.close(fig)

# ============================================================ Figure 2
# values: Table 9 (code/exp10_competitors.py, author's run), p=4, m=5, n=50, N=250
eps = [0.0, 0.10, 0.20, 0.30]
res = {
    "mean $(N)$":           ([0.1256, 0.6174, 1.2293, 1.8080], GREY,  ":",  "x"),
    "median $(N)$":         ([0.1590, 0.2067, 0.3616, 0.6000], GREEN, "-.", "v"),
    "OGK $(N)$":            ([0.1339, 0.1340, 0.8377, 1.8572], "0.35", "--", "D"),
    "Fast-MCD $(N)$":       ([0.1399, 0.1368, 0.3962, 2.0241], BLUE,  "-",  "s"),
    "Fast-MCD $(n$, SRS$)$":([0.3624, 0.3690, 0.9380, 2.1626], BLUE,  "--", "o"),
    "inner-rank $(n)$":     ([0.2207, 0.2274, 0.2546, 0.3570], RED,   "-",  "^"),
}
fig, ax = plt.subplots(figsize=(4.6, 3.1))
for lab, (y, col, ls, mk) in res.items():
    lw = 2.2 if lab.startswith("inner") else 1.3
    ax.plot(eps, y, color=col, ls=ls, marker=mk, ms=4.5, lw=lw, label=lab)
ax.set_yscale("log")
ax.set_xticks(eps)
ax.set_xlabel(r"contamination fraction $\varepsilon$")
ax.set_ylabel("RMSE of the location estimate")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
ax.set_ylim(0.1, 3)
fig.tight_layout()
fig.savefig("fig_competitors.pdf")
plt.close(fig)

# ============================================================ Figure 3
# values: Table 14 (code/exp12_airquality.py), n = 60, 5000 replications
T = {3: 0.4775, 5: 0.6390}
poll = [  # name, |Pearson rho|, |normal-scores rho|, RE m=3 (se), RE m=5 (se)
    ("benzene", 0.982, 0.946, (1.729, 0.050), (2.560, 0.074)),
    ("CO",      0.879, 0.844, (1.525, 0.043), (1.932, 0.054)),
    (r"NO$_x$", 0.656, 0.750, (1.387, 0.041), (1.648, 0.048)),
    (r"NO$_2$", 0.157, 0.163, (1.003, 0.028), (1.044, 0.030)),
]
fig, ax = plt.subplots(figsize=(4.4, 3.0))
r = np.linspace(0, 1, 300)
for m, col in ((3, BLUE), (5, RED)):
    ax.plot(r, 1 / (1 - r ** 2 * T[m]), color=col, lw=1.2,
            label=rf"normal-theory prediction, $m={m}$")
for name, rp, rns, re3, re5 in poll:
    for m, (re_, se), col in ((3, re3, BLUE), (5, re5, RED)):
        lo, hi = sorted((rp, rns))
        ax.plot([lo, hi], [re_, re_], color=col, lw=2.0, alpha=0.55,
                solid_capstyle="butt")
        ax.errorbar((lo + hi) / 2, re_, yerr=1.96 * se, fmt="o", color=col,
                    ms=4, mfc="white", mew=1.1, capsize=2, lw=0.9)
    ax.annotate(name, xy=(max(rp, rns), re5[0]), xytext=(4, 2),
                textcoords="offset points", fontsize=7.5)
ax.set_xlabel(r"sensor--analyser correlation $|\rho|$")
ax.set_ylabel("relative efficiency")
ax.set_xlim(0, 1.05)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("fig_airquality.pdf")
plt.close(fig)
print("wrote fig_theory.pdf, fig_competitors.pdf, fig_airquality.pdf")
