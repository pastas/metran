"""FactorAnalysis class for Metran in Pastas."""

from logging import getLogger

import numpy as np
import scipy.optimize as scopt
from pastas.utils import initialize_logger

logger = getLogger(__name__)
initialize_logger(logger)


class FactorAnalysis:
    """Class to perform a factor analysis for the Pastas Metran model.

    Parameters
    ----------
    maxfactors : int, optional.
        maximum number of factors to select. The default is None.

    Examples
    --------
    A minimal working example of the FactorAnalysis class is shown below:

    >>> fa = FactorAnalysis()
    >>> factors = fa.solve(oseries)
    """

    def __init__(self, maxfactors=None):
        self.maxfactors = maxfactors

    def get_eigval_weight(self):
        """Method to get the relative weight of each eigenvalue.

        Returns
        -------
        numpy.ndarray
            All eigenvalues as a fraction of the sum of eigenvalues.
        """
        return self.eigval / np.sum(self.eigval)

    def solve(self, oseries):
        """Method to perform factor analysis.

        Factor analysis is based on the minres algorithm.
        The number of eigenvalues is determined by MAP test.
        If more than one eigenvalue is used,
        the factors are rotated using orthogonal rotation.

        Parameters
        ----------
        oseries : pandas.DataFrame
            Object containing the time series. The
            series can be non-equidistant.

        Raises
        ------
        Exception
            If no proper factors can be derived from the series.

        Returns
        -------
        factors : numpy.ndarray
            Factor loadings.
        """
        correlation = self._get_correlations(oseries)
        self.eigval, eigvec = self._get_eigval(correlation)

        # Velicer's MAP test
        try:
            nfactors, _ = self._maptest(correlation,
                                        eigvec, self.eigval)
            msg = "Number of factors according to Velicer\'s MAP test: " \
                  + f"{nfactors}"
            logger.info(msg)
            if nfactors == 0:
                nfactors = sum(self.eigval > 1)
                msg = "Number of factors according to Kaiser criterion: " \
                      + f"{nfactors}"
                logger.info(msg)
            if self.maxfactors is not None:
                nfactors = min(nfactors, self.maxfactors)
        except Exception:
            nfactors = 0
        factors = self._minres(correlation, nfactors)

        if ((nfactors > 0) and (factors is not None)
                and (np.count_nonzero(factors) > 0)):
            # factors is not None and does not contain nonzero elements
            if nfactors > 1:
                # perform varimax rotation
                comm = np.zeros(factors.shape[0])
                for i in range(factors.shape[0]):
                    for j in range(nfactors):
                        comm[i] = comm[i] + factors[i, j] ** 2
                    factors[i, :] = factors[i, :] / np.sqrt(comm[i])
                factors = self._rotate(factors[:, :nfactors])
                for i in range(factors.shape[0]):
                    factors[i, :] = factors[i, :] * np.sqrt(comm[i])

            # swap sign if dominant sign is negative
            for j in range(factors.shape[1]):
                facsign = 0
                for i in range(factors.shape[0]):
                    facsign = facsign + factors[i, j]
                if facsign < 0:
                    for i in range(factors.shape[0]):
                        if np.sign(factors[i, j]) != 0:
                            factors[i, j] = -1. * factors[i, j]

            self.factors = np.atleast_2d(factors[:, :nfactors])

            self.fep = 100 * np.sum(self.get_eigval_weight()[:nfactors])

        else:
            msg = "No proper common factors could be derived from series."
            logger.warning(msg)
            self.factors = None

        return self.factors

    @staticmethod
    def _rotate(phi, gamma=1, maxiter=20, tol=1e-6):
        """Internal method to rotate factor loadings.

        Uses varimax, quartimax, equamax, or parsimax rotation.

        Parameters
        ----------
        phi : numpy.ndarray
            Eigenvectors to be rotated
        gamma : float, optional
            Coefficient for rotation. The default is 1.
            Varimax: gamma = 1.
            Quartimax: gamma = 0.
            Equamax: gamma = nfac/2.
            Parsimax: gamma = nvar(nfac - 1)/(nvar + nfac - 2).
        maxiter : integer, optional
            Maximum number of iterations. The default is 20.
        tol : float, optional
            Stop criterion. The default is 1e-6.

        Returns
        -------
        phi_rot : 2-dimensional array
            rotated eigenvectors

        References
        ----------
        Kaiser, H.F. (1958): The varimax criterion for analytic rotation in
        factor analysis. Psychometrika 23: 187–200.
        """
        p, k = phi.shape
        R = np.eye(k)
        d = 0
        for _ in range(maxiter):
            d_old = d
            Lambda = np.dot(phi, R)
            u, s, vh = np.linalg.svd(
                np.dot(phi.T, np.asarray(Lambda) ** 3 - (gamma / p)
                       * np.dot(Lambda, np.diag(
                           np.diag(np.dot(Lambda.T, Lambda))))))
            R = np.dot(u, vh)
            d = np.sum(s)
            if (d_old != 0) and (d / d_old < 1 + tol):
                break

        phi_rot = np.dot(phi, R)
        return phi_rot

    def _minres(self, s, nf, covar=False):
        """Internal method for estimating factor loadings.

        Uses the minimum residuals (minres) algorithm.

        Parameters
        ----------
        s : numpy.ndarray
            Correlation matrix
        nf : integer
            Number of factors
        covar : boolean
            True if S is covar

        Returns
        -------
        loadings : numpy.ndarray
            Estimated factor loadings
        """
        sorg = np.copy(s)
        try:
            ssmc = 1 - 1 / np.diag(np.linalg.inv(s))
            if (not(covar) and np.sum(ssmc) == nf) and (nf > 1):
                start = 0.5 * np.ones(nf, dtype=float)
            else:
                start = np.diag(s) - ssmc
        except:
            return

        bounds = list()
        for _ in range(len(start)):
            bounds.append((0.005, 1))

        res = scopt.minimize(self._minresfun, start, method='L-BFGS-B',
                             jac=self._minresgrad, bounds=bounds,
                             args=(s, nf))

        loadings = self._get_loadings(res.x, sorg, nf)

        return loadings

    @staticmethod
    def _maptest(cov, eigvec, eigval):
        """Internal method to run Velicer's MAP test.

        Determines the number of factors to be used. This method includes
        two variations of the MAP test: the orginal and the revised MAP test.

        Parameters
        ----------
        cov : numpy.ndarray
            Covariance matrix.
        eigvec : numpy.ndarray
            Matrix with columns eigenvectors associated with eigenvalues.
        eigval : numpy.ndarray
            Vector with eigenvalues in descending order.

        Returns
        -------
        nfacts : integer
            Number factors according to MAP test.
        nfacts4 : integer
            Number factors according to revised MAP test.

        References
        ----------
        The original MAP test:

        Velicer, W. F. (1976). Determining the number of components
        from the matrix of partial correlations. Psychometrika, 41, 321-327.

        The revised (2000) MAP test i.e., with the partial correlations
        raised to the 4rth power (rather than squared):

        Velicer, W. F., Eaton, C. A., and Fava, J. L. (2000). Construct
        explication through factor or component analysis: A review and
        evaluation of alternative procedures for determining the number
        of factors or components. Pp. 41-71 in R. D. Goffin and
        E. Helmes, eds., Problems and solutions in human assessment.
        Boston: Kluwer.
        """

        nvars = len(eigval)
        fm = np.array([np.arange(nvars, dtype=float),
                       np.arange(nvars, dtype=float)]).T
        np.put(fm, [0, 1], ((np.sum(np.sum(np.square(cov))) - nvars)
                            / (nvars * (nvars - 1))))
        fm4 = np.copy(fm)
        np.put(fm4, [0, 1],
               ((np.sum(np.sum(np.square(np.square(cov)))) - nvars)
                / (nvars * (nvars - 1))))

        for m in range(nvars - 1):
            biga = np.atleast_2d(eigvec[:, :m + 1])
            partcov = cov - np.dot(biga, biga.T)
            # exit function with nfacts=1 if diag partcov contains negatives
            if np.amin(np.diag(partcov)) < 0:
                return 1, 1
            d = np.diag((1 / np.sqrt(np.diag(partcov))))
            pr = np.dot(d, np.dot(partcov, d))
            np.put(fm, [m + 1, 1], ((np.sum(np.sum(np.square(pr))) - nvars)
                                    / (nvars * (nvars - 1))))
            np.put(fm4, [m + 1, 1], ((np.sum(np.sum(np.square(np.square(pr))))
                                      - nvars) / (nvars * (nvars - 1))))

        minfm = fm[0, 1]
        nfacts = 0
        minfm4 = fm4[0, 1]
        nfacts4 = 0
        for s in range(nvars):
            fm[s, 0] = s
            fm4[s, 0] = s
            if fm[s, 1] < minfm:
                minfm = fm[s, 1]
                nfacts = s
            if fm4[s, 1] < minfm4:
                minfm4 = fm4[s, 1]
                nfacts4 = s
        return nfacts, nfacts4

    @staticmethod
    def _minresfun(psi, s, nf):
        """Function to be minimized in minimum residuals (minres) algorithm.

        Parameters
        ----------
        psi : array
            Vector to be adjusted during optimization
        s : array
            Correlation matrix
        nf : integer
            Number of factors

        Returns
        -------
        obj : array
            objective function defined as sum of residuals
        """
        s2 = np.copy(s)
        np.fill_diagonal(s2, 1 - psi)
        eigval, eigvec = np.linalg.eigh(s2)
        eigval[eigval < np.MachAr().eps] = 100 * np.MachAr().eps
        if nf > 1:
            loadings = np.atleast_2d(np.dot(eigvec[:, :nf],
                                            np.diag(np.sqrt(eigval[:nf]))))
        else:
            loadings = eigvec[:, 0] * np.sqrt(eigval[0])
        model = np.dot(loadings, loadings.T)
        residual = np.square(s2 - model)
        np.fill_diagonal(residual, 0)

        return np.sum(residual)

    def _minresgrad(self, psi, s, nf):
        """Internal method to calculate jacobian of function.

        Jacobian to be minimized in minimum residuals (minres) algorithm.

        Parameters
        ----------
        psi : array
            Vector to be adjusted during optimization.
        s : array
            Correlation matrix.
        nf : integer
            Number of factors.

        Returns
        -------
        jac : array
            Jacobian of minresfun.
        """

        load = self._get_loadings(psi, s, nf)
        g = np.dot(load, load.T) + np.diag(psi) - s
        jac = np.diag(g) / np.square(psi)

        return jac

    @staticmethod
    def _get_loadings(psi, s, nf):
        """Internal method to estimate matrix of factor loadings.

        Based on minimum residuals (minres) algorithm.

        Parameters
        ----------
        psi : numpy.ndarray
            Communality estimate.
        s : numpy.ndarray
            Correlation matrix.
        nf : integer
            Number of factors.

        Returns
        -------
        load : npumy.ndarray
            Estimated factor loadings.
        """
        sc = np.diag(1 / np.sqrt(psi))
        sstar = np.dot(sc, np.dot(s, sc))
        eigval, eigvec = np.linalg.eig(sstar)
        L = eigvec[:, :nf]
        load = np.dot(L, np.diag(np.sqrt(np.maximum(
            np.subtract(eigval[:nf], 1), 0))))
        load = np.dot(np.diag(np.sqrt(psi)), load)
        return load

    @staticmethod
    def _get_correlations(oseries):
        """Internal method to calculate correlations for multivariate series.

        Parameters
        ----------
        oseries : pandas.DataFrame
            Multivariate series

        Returns
        -------
        corr : numpy.ndarray
            Correlation matrix
        """
        corr = np.array(oseries.corr())
        return corr

    @staticmethod
    def _get_eigval(correlation):
        """Internal method to get eigenvalues and eigenvectors.

        Get eigenvalues and eigenvectors based on correlation matrix.

        Parameters
        ----------
        correlation : numpy.ndarray
            Correlation matrix for which eigenvalues
            and eigenvectors need to be derived.

        Raises
        ------
        Exception
            If method results in complex eigenvalues and eigenvectors.

        Returns
        -------
        eigval : numpy.ndarray
            Vector with eigenvalues.
        eigvec : numpy.ndarray
            Matrix with eigenvectors.
        """
        # perform eigenvalue decomposition
        eigval, eigvec = np.linalg.eig(correlation)
        if isinstance(eigval[0], np.complex128):
            msg = "Serial correlation matrix has " + \
                  "complex eigenvalues and eigenvectors. " + \
                  "Factors cannot be estimated for these series."
            logger.error(msg)
            raise Exception(msg)
        # sort eigenvalues and eigenvectors
        evals_order = np.argsort(-eigval)
        eigval = eigval[evals_order]
        eigval[eigval < 0] = 0.
        eigvec = eigvec[:, evals_order]
        eigvec = np.atleast_2d(np.dot(eigvec, np.sqrt(np.diag(eigval))))
        return eigval, eigvec
