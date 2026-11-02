"""Sparse IRLS for Poisson / NB2 log-link GLMs with an offset.

The design matrix is ~95 % dummy zeros, so X'WX is built with scipy.sparse in milliseconds
instead of the dense statsmodels path (which made the NB2 profile likelihood take > 1 h).
Results are checked against statsmodels in run_all.py (max |beta difference| is reported).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import stats
from scipy.special import gammaln


class FastGLM:
    def __init__(self, X: pd.DataFrame, y, offset, alpha=0.0, beta0=None, tol=1e-10, maxit=100):
        self.names = list(X.columns)
        self.Xs = sps.csr_matrix(X.values.astype(float))
        self.y = np.asarray(y, float)
        self.off = np.asarray(offset, float)
        self.alpha = float(alpha)
        self._fit(beta0, tol, maxit)

    # -------------------------------------------------------------- IRLS
    def _fit(self, beta0, tol, maxit):
        Xs, y, off, a = self.Xs, self.y, self.off, self.alpha
        if beta0 is None:
            mu = y + y.mean()
            eta = np.log(mu)
        else:
            eta = Xs @ beta0 + off
            mu = np.exp(eta)
        dev_old = np.inf
        for it in range(maxit):
            w = mu / (1.0 + a * mu)
            z = eta - off + (y - mu) / mu
            XtW = Xs.T.multiply(w).tocsr()
            H = (XtW @ Xs).toarray()
            g = XtW @ z
            beta = np.linalg.solve(H, g)
            eta = Xs @ beta + off
            mu = np.exp(np.clip(eta, -30, 30))
            dev = self._deviance(mu)
            if abs(dev_old - dev) < tol * (abs(dev) + 0.1):
                break
            dev_old = dev
        w = mu / (1.0 + a * mu)
        XtW = Xs.T.multiply(w).tocsr()
        self.H = (XtW @ Xs).toarray()
        self.beta, self.mu, self.n_iter = beta, mu, it + 1
        self.cov_model = np.linalg.inv(self.H)

    def _deviance(self, mu):
        y, a = self.y, self.alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        if a <= 0:
            return 2 * np.sum(t1 - (y - mu))
        return 2 * np.sum(t1 - (y + 1 / a) * np.log((1 + a * y) / (1 + a * mu)))

    # -------------------------------------------------------------- statistics
    @property
    def llf(self):
        y, mu, a = self.y, self.mu, self.alpha
        if a <= 0:
            return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
        n = 1 / a
        return float(np.sum(stats.nbinom.logpmf(y, n, n / (n + mu))))

    @property
    def params(self):
        return pd.Series(self.beta, index=self.names)

    @property
    def fittedvalues(self):
        return pd.Series(self.mu)

    @property
    def df_resid(self):
        return len(self.y) - len(self.beta)

    @property
    def pearson_chi2(self):
        v = self.mu + self.alpha * self.mu ** 2
        return float(np.sum((self.y - self.mu) ** 2 / v))

    @property
    def aic(self):
        return -2 * self.llf + 2 * len(self.beta)

    def predict(self, X, offset):
        return np.exp(sps.csr_matrix(X.values.astype(float)) @ self.beta + np.asarray(offset, float))

    def cluster_cov(self, groups):
        """Sandwich covariance clustered by `groups`, with the usual G/(G-1)*(n-1)/(n-p) correction."""
        g = pd.factorize(np.asarray(groups))[0]
        G = g.max() + 1
        r = (self.y - self.mu) / (1 + self.alpha * self.mu)
        U = self.Xs.multiply(r[:, None]).tocsr()
        Gm = sps.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(G, len(g)))
        Sg = (Gm @ U).toarray()
        meat = Sg.T @ Sg
        n, p = self.Xs.shape
        c = G / (G - 1) * (n - 1) / (n - p)
        return c * self.cov_model @ meat @ self.cov_model

    def summary_frame(self, cov):
        se = np.sqrt(np.diag(cov))
        z = self.beta / se
        return pd.DataFrame({"coef": self.beta, "se": se, "p_value": 2 * stats.norm.sf(np.abs(z)),
                             "CI_low_coef": self.beta - 1.96 * se, "CI_high_coef": self.beta + 1.96 * se},
                            index=self.names)
