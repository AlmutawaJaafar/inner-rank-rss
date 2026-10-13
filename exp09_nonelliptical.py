"""
exp09_nonelliptical.py -- how far does ellipticity have to hold?

Every theorem in Section 4 rests on the polar decomposition
X = mu + R Sigma^{1/2} U with U uniform on the sphere and independent of R.
Two distinct departures are worth separating, and they behave very differently.

  (A) HEAVY TAILS, STILL ELLIPTICAL.  The multivariate t is elliptical, so
      Lemma 1 holds verbatim and the estimator stays unbiased.  Only the
      constant changes: for a general elliptical law with finite second moment,

          RE* = E[D] / E[D_(1:m)],        D = (X-mu)' Sigma^{-1} (X-mu),

      which reduces to p/c_1 in the Gaussian case since E[D] = p.  For the
      multivariate t with nu degrees of freedom, D/p ~ F(p, nu), so
      E[D] = p*nu/(nu-2).  This script checks that prediction, which if it
      holds means the theory is not Gaussian-specific at all.

  (B) SKEWNESS, NOT ELLIPTICAL.  Here the licence fails: E[X | rank r] is no
      longer mu, inner-rank selection pulls the sample toward the mode, and the
      failure mode is a BIAS that does not shrink with n.  This script measures
      how quickly that bias grows with skewness, and at what point it destroys
      the advantage.  The comparison is by mean squared error, not variance,
      since bias is the whole issue.

Run:  python3.10 -u exp09_nonelliptical.py | tee exp09.txt     (about 4 minutes)
"""
import numpy as np
from scipy import stats, integrate
import time

SEED = 20260925


def ar1(p, rho=0.5):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


# ---------------------------------------------------------------- (A) theory
def ED1_generic(dist, m):
    """
    E[D_(1:m)] via the survival integral  E[Y] = int_0^inf P(Y > x) dx  with
    P(min > x) = (1-F(x))^m.  This form is numerically stable for heavy-tailed
    D, where the density form overflows: (1-F)^m decays like x^{-nu m/2}.
    """
    S = lambda x: (1 - dist.cdf(x)) ** m
    a = integrate.quad(S, 0, dist.ppf(0.999), limit=500)[0]
    b = integrate.quad(S, dist.ppf(0.999), np.inf, limit=500)[0]
    return a + b


def theory_t(p, m, nu):
    """RE* = E[D]/E[D_(1:m)] for the multivariate t: D/p ~ F(p,nu)."""
    class Dlaw:
        def pdf(self, x): return stats.f.pdf(x / p, p, nu) / p
        def cdf(self, x): return stats.f.cdf(x / p, p, nu)
        def ppf(self, q): return p * stats.f.ppf(q, p, nu)
    ED = p * nu / (nu - 2)
    return ED / ED1_generic(Dlaw(), m)


def theory_gauss(p, m):
    class Dlaw:
        def pdf(self, x): return stats.chi2.pdf(x, p)
        def cdf(self, x): return stats.chi2.cdf(x, p)
        def ppf(self, q): return stats.chi2.ppf(q, p)
    return p / ED1_generic(Dlaw(), m)


# ------------------------------------------------------------- (A) simulation
def run_elliptical(p, m, n, reps, nu=None, seed=0):
    """nu = None gives the Gaussian case; otherwise a multivariate t_nu."""
    rng = np.random.default_rng([SEED, seed])
    Sigma = ar1(p); chol = np.linalg.cholesky(Sigma); Sinv = np.linalg.inv(Sigma)

    def draw(shape):
        Z = rng.standard_normal((*shape, p)) @ chol.T
        if nu is None:
            return Z
        W = rng.chisquare(nu, size=(*shape, 1)) / nu
        return Z / np.sqrt(W)                      # elliptical t

    Ls, Li, Bi = [], [], []
    for _ in range(reps):
        U = draw((n, m))
        d = np.einsum("...i,ij,...j->...", U, Sinv, U)
        X = U[np.arange(n), np.argmin(d, axis=1)]
        Xs = draw((n,))
        Ls.append(np.sum(Xs.mean(0) ** 2)); Li.append(np.sum(X.mean(0) ** 2))
        Bi.append(X.mean(0))
    Ls = np.array(Ls); Li = np.array(Li)
    return {"RE": Ls.mean() / Li.mean(),
            "bias": float(np.linalg.norm(np.array(Bi).mean(0)))}


# -------------------------------------------------------- (B) skewed marginals
def run_skewed(p, m, n, reps, alpha, seed=0, heavy=False):
    """
    Independent standardised skew-normal marginals (Sigma = I), or lognormal
    when heavy=True.  Ranking uses the true mean and covariance, so any bias is
    attributable to the loss of ellipticity and not to pilot error.
    """
    rng = np.random.default_rng([SEED, 700 + seed])

    def draw(shape):
        if heavy:
            Y = rng.lognormal(0.0, 0.75, size=(*shape, p))
            mu, sd = np.exp(0.75 ** 2 / 2), np.sqrt(
                (np.exp(0.75 ** 2) - 1) * np.exp(0.75 ** 2))
        else:
            Y = stats.skewnorm.rvs(alpha, size=(*shape, p),
                                   random_state=rng.integers(1 << 31))
            d = alpha / np.sqrt(1 + alpha ** 2)
            mu = d * np.sqrt(2 / np.pi)
            sd = np.sqrt(1 - 2 * d ** 2 / np.pi)
        return (Y - mu) / sd                        # mean 0, variance 1

    Ls, Li, Bi = [], [], []
    for _ in range(reps):
        U = draw((n, m))
        d = (U ** 2).sum(axis=2)
        X = U[np.arange(n), np.argmin(d, axis=1)]
        Xs = draw((n,))
        Ls.append(np.sum(Xs.mean(0) ** 2)); Li.append(np.sum(X.mean(0) ** 2))
        Bi.append(X.mean(0))
    Ls = np.array(Ls); Li = np.array(Li)
    bias = float(np.linalg.norm(np.array(Bi).mean(0)))
    relse = np.sqrt(Ls.var(ddof=1) / (len(Ls) * Ls.mean() ** 2) +
                    Li.var(ddof=1) / (len(Li) * Li.mean() ** 2))
    return {"relse": relse, "RE_mse": Ls.mean() / Li.mean(), "bias": bias,
            "bias_per_se": bias / np.sqrt(p / n)}   # SE of the SRS mean is sqrt(p/n)


if __name__ == "__main__":
    t0 = time.time(); p, m, n = 4, 5, 50

    print("=" * 74)
    print("(A) HEAVY TAILS: the multivariate t is elliptical, so the theory")
    print("    should hold with RE* = E[D]/E[D_(1:m)] and zero bias.")
    print("=" * 74)
    print(f"{'law':>12}{'E[D]':>9}{'RE* theory':>12}{'RE observed':>13}{'bias':>9}")
    r = run_elliptical(p, m, n, 6000, nu=None, seed=1)
    print(f"{'Gaussian':>12}{p:>9.3f}{theory_gauss(p,m):>12.4f}"
          f"{r['RE']:>13.4f}{r['bias']:>9.4f}")
    for nu in (10, 5, 3):
        th = theory_t(p, m, nu)
        r = run_elliptical(p, m, n, 6000, nu=nu, seed=nu)
        print(f"{'t_'+str(nu):>12}{p*nu/(nu-2):>9.3f}{th:>12.4f}"
              f"{r['RE']:>13.4f}{r['bias']:>9.4f}")

    print("\n" + "=" * 74)
    print("(B) SKEWNESS: ellipticity fails, E[X|rank r] != mu, and the")
    print("    inner-rank sample is pulled toward the mode.")
    print("    'bias/SE' expresses the bias in units of the SRS standard error,")
    print("    sqrt(tr Sigma / n) = sqrt(p/n) = %.3f here." % np.sqrt(p / n))
    print("=" * 74)
    print(f"{'population':>22}{'skewness':>10}{'RE (MSE)':>10}{'bias':>9}{'bias/SE':>9}")
    for alpha in (0.0, 1.0, 2.0, 5.0, 10.0):
        d = alpha / np.sqrt(1 + alpha ** 2)
        g1 = ((4 - np.pi) / 2) * (d * np.sqrt(2 / np.pi)) ** 3 / \
             (1 - 2 * d ** 2 / np.pi) ** 1.5
        r = run_skewed(p, m, n, 4000, alpha, seed=int(10 * alpha))
        print(f"{'skew-normal a=%.0f' % alpha:>22}{g1:>10.3f}"
              f"{r['RE_mse']:>10.3f}{r['bias']:>9.4f}{r['bias_per_se']:>9.2f}")
    r = run_skewed(p, m, n, 4000, 0.0, seed=99, heavy=True)
    print(f"{'lognormal(0,0.75)':>22}{3.26:>10.3f}"
          f"{r['RE_mse']:>10.3f}{r['bias']:>9.4f}{r['bias_per_se']:>9.2f}")

    print(f"\nelapsed {time.time()-t0:.0f}s")
