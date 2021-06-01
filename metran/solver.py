"""This module contains the  solver that is available for Pastas Metran.

All solvers inherit from the BaseSolver class, which contains methods to
obtain the object function value and numerical approximation of the
parameter covariance matrix.


To solve a model the following syntax can be used:

>>> mt.solve(solver=ps.LmfitSolve)
"""

from logging import getLogger

import numpy as np
from pandas import DataFrame
from pastas.utils import initialize_logger
from scipy.optimize import approx_fprime, minimize

logger = getLogger(__name__)
initialize_logger(logger)


class BaseSolver:
    _name = "BaseSolver"
    __doc__ = """All solver instances inherit from the BaseSolver class.

    Attributes
    ----------
    mt : Metran instance

    """

    def __init__(self, mt, **kwargs):
        self.mt = mt
        # Initialize objects
        self.pcov = None
        self.pcor = None
        self.nfev = None
        self.result = None

    def objfunction(self, p, callback):
        """Method to get objective function used by solver.

        Parameters
        ----------
        p : type required for callback function
            Parameters to be coverted by callback function into
            proper type and format.
        callback: ufunc
            Function that is called after each iteration.
            The parameters are provided to the func,
            e.g. "callback(parameters)".

        Returns
        -------
        obj : float
            Objective function value.
        """
        if callback is not None:
            p = callback(p)
        obj = self.mt.get_mle(p)
        return obj

    def _get_covariance(self, x0, f, callback, epsilon=None, diff="forward"):
        """Estimate covariance matrix of parameter estimates.

        Uses a numerical approximation to the Hessian matrix of cost
        function at location x0.

        Parameters
        ----------
        x0 : array_like
            Parameter values at objective function minimum
        f : callable
            The objective function
        callback : ufunc
            Function that is called after each iteration.
            The parameters are provided to the func,
            e.g. "callback(parameters)".
        epsilon : float, optional
            Pertubation in finite difference scheme.
            The default is EPS ** (1. / 4).
            If finite difference approximation results in non-positive
            values on the diagonal of the covariance matrix,
            epsilon will be increase by factor 10.
        diff : str, optional
            forward or central difference scheme. The default is "forward"

        Returns
        -------
        cov : numpy.ndarray
            covariance matrix of parameter estimates
        """
        n = x0.shape[0]
        if epsilon is None:
            EPS = np.MachAr().eps
            epsilon = EPS ** (1. / 4)

        cov_ok = False
        while not(cov_ok or epsilon > 1000. * epsilon):
            # allocate space for the hessian
            hessian = np.zeros((n, n))
            # the next loop fill the hessian matrix
            xx = x0
            if diff == "forward":
                f0 = approx_fprime(x0, f, epsilon, callback)
            for j in range(n):
                xx0 = xx[j]
                d = epsilon * max(np.abs(xx0), 0.1)
                # forward difference
                xx[j] = xx0 + d
                ff = approx_fprime(xx, f, epsilon, callback)
                if diff == "central":
                    # backward difference
                    xx[j] = xx0 - d
                    fb = approx_fprime(xx, f, epsilon, callback)
                    hessian[:j + 1, j] = (ff[:j + 1] - fb[:j + 1]) / (2 * d)
                else:
                    hessian[:j + 1, j] = (ff[:j + 1] - f0[:j + 1]) / d
                hessian[j, :j + 1] = hessian[:j + 1, j]
                # restore initial value of x0
                xx[j] = xx0

            # try to calculate covariance matrix from Hessian
            if not np.isnan(hessian).any():
                cov = np.linalg.pinv(hessian)
                if np.amin(np.diag(cov)) <= 0:
                    # find nearest positive semi-definite matrix
                    try:
                        cov = np.linalg.pinv(self._nearPSD(hessian))
                    except Exception as e:
                        logger.debug(f"Could not calculate 'cov': {e}")
                if np.amin(np.diag(cov)) > 0:
                    cov_ok = True

            if not(cov_ok):
                epsilon *= 10.

        return cov

    @staticmethod
    def _get_correlations(pcov):
        """Internal method to obtain the parameter correlations.

        Parameter correlations are derived from the covariance matrix.

        Parameters
        ----------
        pcov: pandas.DataFrame
            n x n Pandas DataFrame with the covariances.

        Returns
        -------
        pcor: pandas.DataFrame
            n x n Pandas DataFrame with the correlations.
        """
        pcor = pcov.loc[pcov.index, pcov.index].copy()

        for i in pcor.index:
            for j in pcor.columns:
                pcor.loc[i, j] = pcov.loc[i, j] / \
                    np.sqrt(pcov.loc[i, i] * pcov.loc[j, j])
        return pcor

    @staticmethod
    def _nearPSD(A, epsilon=0):
        """Get the nearest Positive Semi-Definite matrix of A.

        Parameters
        ----------
        A : numpy.ndarray
            square matrix
        epsilon : float, optional
            cut-off value for eigenvalues. The default is 0.

        Returns
        -------
        out : numpy.ndarray
           Nearest Positive Semi-Definite matrix of A.
        """
        n = A.shape[0]
        eigval, eigvec = np.linalg.eig(A)
        val = np.matrix(np.maximum(eigval, epsilon))
        vec = np.matrix(eigvec)
        with np.errstate(divide='ignore', invalid='ignore'):
            T = 1 / (np.multiply(vec, vec) * val.T)
            T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
            B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
        out = B * B.T
        return out


class ScipySolve(BaseSolver):
    """Solver based on Scipy's least_squares method [scipy_ref]_.

    This class is the default solver class in the Metran solve method.

    Parameters
    ----------
    mt : Metran instance
    **kwargs : dict, optional
        All keyword arguments will be passed onto minimization method
        from the solver.

    Examples
    --------

    >>> mt.solve(solver=ps.ScipySolve)

    References
    ----------
    .. [scipy_ref] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    """
    _name = "ScipySolve"

    def __init__(self, mt, **kwargs):
        BaseSolver.__init__(self, mt=mt, **kwargs)

    def solve(self, method="l-bfgs-b", **kwargs):
        """Method to run solver and optimize parameters.

        Parameters
        ----------
        method : str, optional
            Name of the fitting method to use. The default is "l-bfgs-b".
        **kwargs : dict, optional
            All keyword arguments will be passed onto minimization method
            from the solver.

        Returns
        -------
        success : boolean
            True if optimization routine terminated successfully.
        params : lmfit.Parameters instance
            Ordered dictionary of Parameter objects.
        """

        # Deal with the parameters
        self.vary = self.mt.parameters.vary.values.astype(bool)
        self.initial = self.mt.parameters.initial.values.copy()
        parameters = self.mt.parameters.loc[self.vary]
        bounds = [(b[0], b[1])
                  for b in parameters.loc[:, ["pmin", "pmax"]].values]

        # Create the Minimizer object and minimize
        self.result = minimize(fun=self.objfunction,
                               method=method,
                               x0=parameters.initial.values,
                               bounds=bounds,
                               args=(self._array_todict,),
                               **kwargs)

        _stderr = np.zeros(parameters.shape[0]) * np.NaN
        if hasattr(self.result, "hess_inv"):
            pcov = self.result.hess_inv.todense()
            _stderr = np.sqrt(np.diag(pcov))
        if np.isnan(_stderr).any():
            # calculate covariance matrix using finite differences
            pcov = self._get_covariance(self.result.x, self.objfunction,
                                        self._array_todict)
            _stderr = np.sqrt(np.diag(pcov))

        optimal = self.initial
        optimal[self.vary] = self.result.x
        stderr = np.zeros(len(optimal)) * np.NaN
        stderr[self.vary] = _stderr

        # Set all parameter attributes
        names = parameters.index.values
        self.pcov = DataFrame(pcov, index=names, columns=names)
        self.pcor = self._get_correlations(self.pcov)

        # Set all optimization attributes
        self.nfev = self.result.nfev
        self.aic = 2 * parameters.shape[0] + self.result.fun
        self.obj_func = self.result.fun

        if hasattr(self.result, "success"):
            success = self.result.success
        else:
            success = True

        return success, optimal, stderr

    def _array_todict(self, p):
        """Update parameter dictionary with array of varying parameters.

        Parameters
        ----------
        p : array_like
            Model parameters (varying)

        Returns
        -------
        dict
            Model parameters as input for objective function
        """
        par = self.initial
        par[self.vary] = p
        return par


class LmfitSolve(BaseSolver):
    """Class for solving the model using the LmFit solver [LM]_.

     Lmfit is basically a wrapper around the scipy solvers, adding some
     functionality for boundary conditions.

    Parameters
    ----------
    mt : Metran instance
    **kwargs : dict, optional
        All keyword arguments will be passed onto minimization method
        from the solver.

    Examples
    --------

    >>> mt.solve(solver=ps.LmfitSolve)

    References
    ----------
    .. [LM] https://github.com/lmfit/lmfit-py/
    """
    _name = "LmfitSolve"

    def __init__(self, mt, **kwargs):
        try:
            # Import Lmfit here, so it is no dependency
            global lmfit
            import lmfit as lmfit
        except ImportError:
            msg = "lmfit not installed. Please install lmfit first."
            logger.error(msg)
            raise ImportError(msg)

        self.mt = mt

        # Initialize objects
        self.pcov = None
        self.pcor = None
        self.nfev = None
        self.result = None

    def solve(self, method="lbfgsb", **kwargs):
        """Method to run solver and optimize parameters.

        Parameters
        ----------
        method : str, optional
            Name of the fitting method to use. The default is "lbfgsb".
        **kwargs : dict, optional
            All keyword arguments will be passed onto minimization method
            from the solver.

        Returns
        -------
        success : boolean
            True if optimization routine terminated successfully.
        params : lmfit.Parameters instance
            Ordered dictionary of Parameter objects.
        """

        # Deal with the parameters
        parameters = lmfit.Parameters()
        self.vary = self.mt.parameters.vary.values.astype(bool)
        self.initial = self.mt.parameters.initial.values.copy()
        p = self.mt.parameters.loc[:, ['initial', 'pmin', 'pmax', 'vary']]
        for k in p.index:
            pp = np.where(p.loc[k].isnull(), None, p.loc[k])
            if method == "lbfgsb":
                parameters.add(k, value=pp[0], min=None, max=None, vary=pp[3])
            else:
                parameters.add(k, value=pp[0], min=pp[1], max=pp[2],
                               vary=pp[3])

        if method == "lbfgsb":
            bounds = [(b[0], b[1]) for b in
                      self.mt.parameters.loc[:, ['pmin', 'pmax']].values]
            kwargs['bounds'] = bounds

        # Create the Minimizer object and minimize
        self.mini = lmfit.Minimizer(userfcn=self.objfunction,
                                    params=parameters,
                                    scale_covar=False,
                                    fcn_args=(self._lmfit_todict,),
                                    **kwargs)
        self.result = self.mini.minimize(method=method)

        optimal = np.array([p.value for p in self.result.params.values()])

        # Set all parameter attributes
        stderr = np.zeros(len(optimal)) * np.NaN
        pcov = None
        if hasattr(self.result, "covar"):
            if self.result.covar is not None:
                pcov = self.result.covar
                stderr = np.sqrt(np.diag(pcov))
        if pcov is None:
                # calculate covariance matrix using finite differences
            pcov = self._get_covariance(optimal, self.objfunction,
                                        self._array_todict)
        stderr[self.vary] = np.sqrt(np.diag(pcov))

        names = self.result.var_names
        self.pcov = DataFrame(pcov, index=names, columns=names)
        self.pcor = self._get_correlations(self.pcov)

        # Set all optimization attributes
        self.nfev = self.result.nfev
        self.obj_func = self.objfunction(optimal, self._array_todict)
        self.aic = 2 * p[p["vary"]].shape[0] + self.obj_func

        if hasattr(self.result, "success"):
            success = self.result.success
        else:
            success = True

        return success, optimal, stderr

    @staticmethod
    def _lmfit_todict(p):
        """Convert lmfit.Parameter instance to dictionary.

        Parameters
        ----------
        p : lmfit.Parameter instance
            Model parameters

        Returns
        -------
        dict
            Model parameters as input for objective function
        """
        return p.valuesdict()

    def _array_todict(self, p):
        """Update parameter dictionary with array of varying parameters.

        Parameters
        ----------
        p : array_like
            Model parameters (varying)

        Returns
        -------
        dict
            Model parameters as input for objective function
        """
        par = self.initial
        par[self.vary] = p
        return par
