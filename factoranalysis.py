# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 08:38:33 2021

@author: Wilbert Berendrecht
"""

from logging import getLogger
import numpy as np
import scipy.optimize as scopt
from pastas.utils import validate_name


class FactorAnalysis:
    """Class that performs a factor analysis for the Pastas Metran model.

    Parameters
    ----------
    oseries: list of pandas.Series or pastas.TimeSeries
        list of pandas Series objects containing the time series. The
        series can be non-equidistant.
    name: str, optional
        String with the name of the model, used in plotting and saving.
    metadata: dict, optional
        Dictionary containing metadata of the oseries, passed on the to
        oseries when creating a pastas TimeSeries object. hence,
        ml.oseries.metadata will give you the metadata.

    Returns
    -------
    ml: pastas.metran.Metran
        Pastas Metran instance.

    Examples
    --------
    A minimal working example of the Model class is shown below:

    >>> oseries = pd.Series([1,2,1], index=pd.to_datetime(range(3), unit="D"))
    >>> ml = Model(oseries)

    """

    def __init__(self, oseries, name=None):#, metadata=None):

        self.logger = getLogger(__name__)

        self.oseries = oseries

        if name is None:
            name = 'Cluster'
        self.name = validate_name(name)

        # Default settings for factor analysis
        self.settings = {
            "max_cdf": 3,
        }

    def get_specificity(self):
        specificity = []
        dim = self.factors.shape[0]
        for n in range(dim):
            specificity.append(1. - np.sum(np.square(self.factors[n,:])))
        return specificity

    def get_eigval_weight(self):
        return self.eigval / np.sum(self.eigval)

    def solve(self):
        """
        Factor analysis
            Performs factor analysis using minres algorithm.
            Number of eigenvalues is determined by MAP test.
            If more than one eigenvalue is used, the factors are rotated using orthogonal rotation.

        Parameters
        ----------

        Returns
        -------

        """

        correlation = self.get_correlations()
        self.eigval, eigvec = self.get_eigval(correlation)

        # Velicer's MAP test
        try:
            nfactors, nfactors4 = self._maptest(correlation,
                                                eigvec, self.eigval)
            nfactors = max(nfactors, 1)
        except:
            nfactors = 1
        factors = self._minres(correlation, nfactors)

        if (factors is not None) and (np.count_nonzero(factors) > 0):
            # factors is not None and does not contain nonzero elements
            if nfactors > 1:
                # perform varimax rotation
                comm = np.zeros(factors.shape[0])
                for i in range(factors.shape[0]):
                    for j in range(nfactors):
                        comm[i] = comm[i] + factors[i, j]**2
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

            self.factors = np.matrix(factors[:, :nfactors])

            msg = "Number of factors according to Velicer\'s MAP test: " + \
                  f"{nfactors}"
            self.logger.info(msg)

            fep = 100*np.sum(self.get_eigval_weight()[:nfactors])
            msg = "Percentage explained by these factors:: " + \
                  f"{fep}"
            self.logger.info(msg)

        else:
            msg = "Metran: factor analysis did not result in proper " + \
                  "common factors and calcutions have been interupted."
            self.logger.error(msg)

        return self.factors

    @staticmethod
    def _rotate(phi, gamma=1, maxiter=20, tol=1e-6):
        """
        Rotate factor loadings
            using varimax, quartimax, equamax, or parsimax rotation

        Parameters
        ----------
        phi : 2-dimensional array [nvar, nfac]
            eigenvectors to be rotated
        gamma : float
            coefficient for rotation (optional);
            varimax: gamma = 1;
            quartimax: gamma = 0;
            equamax: gamma = nfac/2;
            parsimax: gamma = nvar(nfac - 1)/(nvar + nfac - 2)
        maxiter : integer
            maximum number of iterations
        tol : float
            stop criterion

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
        d=0
        for _ in range(maxiter):
            d_old = d
            Lambda = np.dot(phi, R)
            u, s, vh = np.linalg.svd(np.dot(phi.T, np.asarray(Lambda)**3
                                     - (gamma/p) * np.dot(Lambda, np.diag(
                                     np.diag(np.dot(Lambda.T, Lambda))))))
            R = np.dot(u, vh)
            d = np.sum(s)
            if (d_old != 0) and (d / d_old < 1 + tol):
                break

        phi_rot = np.dot(phi, R)

        return phi_rot

    def _minres(self, s, nf, covar=False):
        """
        Minimum residuals (minres) algorithm
        for estimating factor loadings

        Parameters
        ----------
        s : array
            correlation matrix
        nf : integer
            number of factors
        covar : boolean
            True if S is covar

        Returns
        -------
        loadings : array
            estimated factor loadings

        """

        sorg = np.copy(s)
        try:
            ssmc =  1 - 1/np.diag(np.linalg.inv(s))
            if (not(covar) and np.sum(ssmc) == nf) and (nf > 1):
                start = 0.5 * np.ones(nf, dtype=float)
            else:
                start = np.diag(s) - ssmc
        except:
            return

        bounds = list()
        for i in range(len(start)):
            bounds.append((0.005, 1))

        res = scopt.minimize(self._minresfun, start, method='L-BFGS-B',
                            jac=self._minresgrad, bounds=bounds, args=(s, nf))

        loadings = self._get_loadings(res.x, sorg, nf)

        return loadings

    def _maptest(cov, eigvec, eigval):
        """
        Velicer's MAP test

        Parameters
        ----------
        cov : array
            covariance matrix
        eigvec : array
            matrix with columns eigenvectors associated with eigenvalues
        eigval : array
            vector with eigenvalues in descending order

        Returns
        -------
        nfacts : integer
            number factors according to MAP test
        nfacts4 : integer
            number factors according to revised MAP test


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
                            / (nvars * (nvars-1))))
        fm4 = np.copy(fm)
        np.put(fm4, [0, 1],
               ((np.sum(np.sum(np.square(np.square(cov)))) - nvars)
                / (nvars * (nvars-1))))

        for m in range(nvars-1):
            biga = np.matrix(eigvec[:, :m+1])
            partcov = cov - np.dot(biga, biga.T)
            # exit function with nfacts=1 if diag partcov contains negatives
            if np.amin(np.diag(partcov)) < 0:
                return 1, 1
            d = np.diag((1 / np.sqrt(np.diag(partcov))))
            pr = np.dot(d, np.dot(partcov, d))
            np.put(fm, [m+1, 1], ((np.sum(np.sum(np.square(pr))) - nvars)
                                  / (nvars * (nvars-1))))
            np.put(fm4, [m+1, 1], ((np.sum(np.sum(np.square(np.square(pr))))
                                    - nvars) / (nvars * (nvars-1))))

        minfm = fm[0, 1]
        nfacts = 0
        minfm4 = fm4[0, 1]
        nfacts4 = 0
        for s in range(nvars):
            fm[s, 0]  = s
            fm4[s, 0] = s
            if fm[s, 1] < minfm:
                minfm  = fm[s, 1]
                nfacts  = s
            if fm4[s, 1] < minfm4:
                minfm4 = fm4[s, 1]
                nfacts4 = s

        return nfacts, nfacts4

    @staticmethod
    def _minresfun(psi, s, nf):
        """
        Function to be minimized in minimum residuals (minres) algorithm

        Parameters
        ----------
        psi : array
            vector to be adjusted during optimization
        s : array
            correlation matrix
        nf : integer
            number of factors

        Returns
        -------
        obj : array
            objective function defined as sum of residuals

        """

        s2 = np.copy(s)
        np.fill_diagonal(s2, 1-psi)
        eigval, eigvec = np.linalg.eigh(s2)
        eigval[eigval < np.MachAr().eps] = 100*np.MachAr().eps
        if nf > 1:
            loadings = np.matrix(np.dot(eigvec[:, :nf],
                                        np.diag(np.sqrt(eigval[:nf]))))
        else:
            loadings = eigvec[:, 0] * np.sqrt(eigval[0])
        model = np.dot(loadings,loadings.T)
        residual = np.square(s2 - model)
        np.fill_diagonal(residual, 0)

        return np.sum(residual)

    def _minresgrad(self, psi, s, nf):
        """
        Jacobian of function to be minimized in minimum
        residuals (minres) algorithm

        Parameters
        ----------
        psi : array
            vector to be adjusted during optimization
        s : array
            correlation matrix
        nf : integer
            number of factors

        Returns
        -------
        jac : array
            jacobian of minresfun

        """

        load = self._get_loadings(psi, s, nf)
        g = np.dot(load, load.T) + np.diag(psi) - s
        jac = np.diag(g) / np.square(psi)

        return jac

    @staticmethod
    def _get_loadings(psi, s, nf):
        """
        Estimate matrix of factor loadings
        based on minimum residuals (minres) algorithm

        Parameters
        ----------
        psi : array [nf]
            communality estimate
        s : array
            correlation matrix
        nf : integer
            number of factors

        Returns
        -------
        load : array
            estimated factor loadings

        """
        sc = np.diag(1 / np.sqrt(psi))
        sstar = np.dot(sc, np.dot(s, sc))
        eigval, eigvec = np.linalg.eig(sstar)
        L = eigvec[:, :nf]
        load = np.dot(L, np.diag(np.sqrt(np.maximum(
            np.subtract(eigval[:nf], 1), 0))))

        load = np.dot(np.diag(np.sqrt(psi)), load)

        return load

    def get_correlations(self, oseries=None):
        """
        Method to get correlations of multivariate series

        Parameters
        ----------
        oseries : pandas DataFrame, optional
            multivariate series

        Returns
        -------
        corr : numpy ndarray
            correlation matrix

        """
        if oseries is None:
            oseries = self.oseries
        corr = np.array(oseries.corr())

        return corr

    def get_eigval(self, correlation):
        # perform eigenvalue decomposition
        eigval, eigvec = np.linalg.eig(correlation)
        if isinstance(eigval[0], np.complex128):
            msg = "Metran: Correlation matrix has "+ \
                  "complex eigenvalues and eigenvectors."
            self.logger.error(msg)
            return

        # sort eigenvalues and eigenvectors
        evals_order = np.argsort(-eigval)
        eigval = eigval[evals_order]
        eigval[eigval < 0] = 0.
        eigvec = eigvec[:, evals_order]

        eigvec = np.matrix(np.dot(eigvec, np.sqrt(np.diag(eigval))))

        return eigval, eigvec
