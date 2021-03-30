"""
This module contains the  solver that is available for Pastas Metran.

All solvers inherit from the BaseSolver class, which contains general method
for selecting the correct time series to misfit and options to weight the
residuals or noise series.


To solve a model the following syntax can be used:

>>> ml.solve(solver=ps.LmfitSolve)

"""

from logging import getLogger

import numpy as np
from pandas import DataFrame
from statsmodels.stats.moment_helpers import cov2corr
from copy import deepcopy

logger = getLogger(__name__)


class BaseSolver:
    _name = "BaseSolver"
    __doc__ = """All solver instances inherit from the BaseSolver class.

    Attributes
    ----------
    model: pastas.Model instance
    pcor: pandas.DataFrame
        Pandas DataFrame with the correlation between the optimized parameters.
    pcov: pandas.DataFrame
        Pandas DataFrame with the correlation between the optimized parameters.
    nfev: int
        Number of times the model is called during optimization.
    result: object
        The object returned by the minimization method that is used. It depends
        on the solver what is actually returned.

    """

    def __init__(self, ml, pcov=None, nfev=None, obj_func=None, **kwargs):
        self.ml = ml
        self.pcov = pcov  # Covariances of the parameters
        if pcov is None:
            self.pcor = None  # Correlation between parameters
        else:
            self.pcor = self._get_correlations(pcov)
        self.nfev = nfev  # number of function evaluations
        self.obj_func = obj_func
        self.result = None  # Object returned by the optimization method

    def ci_simulation(self, n=1000, alpha=0.05, **kwargs):
        """Method to calculate the confidence interval for the simulation.

        Returns
        -------

        Notes
        -----
        The confidence interval shows the uncertainty in the simulation due
        to parameter uncertainty. In other words, there is a 95% probability
        that the true best-fit line for the observed data lies within the
        95% confidence interval.

        """
        return self._get_confidence_interval(func=self.ml.simulate, n=n,
                                             alpha=alpha, **kwargs)

    def _get_realizations(self, func, n=None, name=None, **kwargs):
        """Internal method to obtain n number of parameter realizations."""
        if name:
            kwargs["name"] = name

        parameter_sample = self._get_parameter_sample(n=n, name=name)
        data = {}

        for i, p in enumerate(parameter_sample):
            data[i] = func(p=p, **kwargs)

        return DataFrame.from_dict(data, orient="columns")

    def _get_confidence_interval(self, func, n=None, name=None, alpha=0.05,
                                 **kwargs):
        """Internal method to obtain a confidence interval."""
        q = [alpha / 2, 1 - alpha / 2]
        data = self._get_realizations(func=func, n=n, name=name, **kwargs)

        return data.quantile(q=q, axis=1).transpose()

    def _get_parameter_sample(self, name=None, n=None):
        """Internal method to obtain a parameter sets.

        Parameters
        ----------
        n: int, optional
            Number of random samples drawn from the bivariate normal
            distribution.
        name: str, optional
            Name of the stressmodel or model component to obtain the
            parameters for.

        Returns
        -------
        numpy.ndarray
            Numpy array with N parameter samples.

        """
        p = self.ml.get_parameters(name=name)
        pcov = self._get_covariance_matrix(name=name)

        if name is None:
            parameters = self.ml.parameters
        else:
            parameters = self.ml.parameters.loc[
                self.ml.parameters.name == name]

        pmin = parameters.pmin.fillna(-np.inf).values
        pmax = parameters.pmax.fillna(np.inf).values

        if n is None:
            # only use parameters that are varied.
            n = int(10 ** parameters.vary.sum())

        samples = np.zeros((0, p.size))

        # Start truncated multivariate sampling
        it = 0
        while samples.shape[0] < n:
            s = np.random.multivariate_normal(p, pcov, size=(n,),
                                              check_valid="ignore")
            accept = s[(np.min(s - pmin, axis=1) >= 0) &
                       (np.max(s - pmax, axis=1) <= 0)]
            samples = np.concatenate((samples, accept), axis=0)

            # Make sure there's no endless while loop
            if it > 10:
                break
            else:
                it += 1

        return samples[:n, :]

    def _get_covariance_matrix(self, name=None):
        """Internal method to obtain the covariance matrix from the model.

        Parameters
        ----------
        name: str, optional
            Name of the stressmodel or model component to obtain the
            parameters for.

        Returns
        -------
        pcov: pandas.DataFrame
            Pandas DataFrame with the covariances for the parameters.

        """
        if name:
            index = self.ml.parameters.loc[self.ml.parameters.loc[:,
                                           "name"] == name].index
        else:
            index = self.ml.parameters.index

        pcov = self.pcov.reindex(index=index, columns=index).fillna(0)

        return pcov

    @staticmethod
    def _get_correlations(pcov):
        """Internal method to obtain the parameter correlations from the
        covariance matrix.

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

    def to_dict(self):
        data = {
            "name": self._name,
            "pcov": self.pcov,
            "nfev": self.nfev,
            "obj_func": self.obj_func
        }
        return data


class LmfitSolve(BaseSolver):
    _name = "LmfitSolve"

    def __init__(self, ml, pcov=None, nfev=None, **kwargs):
        """Solving the model using the LmFit solver [LM]_.

         This is basically a wrapper around the scipy solvers, adding some
         cool functionality for boundary conditions.

        References
        ----------
        .. [LM] https://github.com/lmfit/lmfit-py/
        """
        try:
            global lmfit
            import lmfit as lmfit  # Import Lmfit here, so it is no dependency
        except ImportError:
            msg = "lmfit not installed. Please install lmfit first."
            raise ImportError(msg)
        BaseSolver.__init__(self, ml=ml, pcov=pcov, nfev=nfev, **kwargs)

    def solve(self, callback=None, method="bfgs", **kwargs):

        # Deal with the parameters
        parameters = lmfit.Parameters()
        p = self.ml.parameters.loc[:, ['initial', 'pmin', 'pmax', 'vary']]
        initial = {}
        for k in p.index:
            pp = np.where(p.loc[k].isnull(), None, p.loc[k])
            initial[k] = pp[0]
            parameters.add(k, value=pp[0], min=pp[1], max=pp[2], vary=pp[3])

        # Create the Minimizer object and minimize
        self.mini = lmfit.Minimizer(userfcn=self.objfunction,
                                    params=parameters,
                                    scale_covar=False,
                                    fcn_args=(callback,),
                                      **kwargs)

        self.result = self.mini.minimize(method=method)

        # scale covariances by scale parameter
        kf = self.ml.simulate(self.result.params)
        sigma = kf.get_scale()
        print (sigma)
        self.result.params = self.ml.scale_covariances(self.result.params,
                                                       sigma)

        # calculate covariance matrix using finite differences
        self.result = self.stdcorr(self.result, self.objfunction,
                                    (callback,))

        # Set all parameter attributes
        pcov = None
        if hasattr(self.result, "covar"):
            if self.result.covar is not None:
                pcov = self.result.covar

        names = self.result.var_names
        self.pcov = DataFrame(pcov, index=names, columns=names)
        self.pcor = self._get_correlations(self.pcov)

        # Set all optimization attributes
        self.nfev = self.result.nfev
        self.obj_func = self.result.chisqr

        if hasattr(self.result, "success"):
            success = self.result.success
        else:
            success = True

        return success, self.result.params

    def objfunction(self, p, callback):
        kf = self.ml.simulate(p.valuesdict())
        obj = kf.get_mle()

        return obj

    def stdcorr(self, optresult, f, fcn_args, epsilon=None, cutoff=True,
                diff='forward'):
        """
        Estimate correlation matrix and standard error of parameter estimates
        using a numerical approximation to the Hessian matrix of cost function
        at location x0. This function is applied as post-processing of
        optimization result of :func:`lmfit.Minimize`

        Parameters
        ----------
        optresult : Minimizer object
            output from :func:`lmfit.Minimize` containing optimization results
        f : function object
            cost function
        arglist : list
            arguments to be passed to function `f`
        epsilon : float (optional)
            pertubation in finite difference scheme,
            default = :math:`\\epsilon^{1/4}\\mathrm{max}(x_0,0.1)`.
            If finite difference approximation results in `NaN` for
            standard deviation, epsilon will be increase with factor 10.
        cutoff : boolean (optional)
            determine whether standard deviation for `res_r`
            (measurement variance) will be set to `None` if parameter value
            is close to zero (normality assumption is violated)
        diff : string
            forward or central difference scheme (default: forward)

        Returns
        -------
        optresult : float
            coefficient of determination

        Notes
        -----
        The central difference approximation of the hessian matrix is:

        .. math::
            \\frac{\\partial^2 \\mathcal{L}}{\\partial\\psi_i \\partial\\psi_j}
            = \\frac{\\mathcal{L}(\\psi+ e_i+ e_j)
                     -\\mathcal{L}(\\psi+ e_i- e_j)
            -\\mathcal{L}(\\psi- e_i+ e_j)
            +\\mathcal{L}(\\psi-e_i-e_j)}{4e_ie_j}

        References
        ----------

        Ridout, M.S. (2009), Statistical applications of the complex-step
        method of numerical differentiation. The American Statistician,
        63, 66-74

        """
        # store initial parameter values in local variable initvalue,
        #  to prevent them from being overwritten
        initvalue = []
        for name in optresult.params:
            if optresult.params[name].vary:
                initvalue.append(optresult.params[name].init_value)

        # make a copy of optresult object and use this copy as local object
        optlocal = deepcopy(optresult)

        params = optlocal.params
        # update parameters with expressions
        params.update_constraints()

        # set stderr of (non-vary and non-expr) to None
        for name in optresult.params:
            if (not(optresult.params[name].vary)
                and optresult.params[name].expr is None):
                optresult.params[name].stderr = None

       # initialize local variables and objects
        x0 = []
        x0name = []
        has_expr = False
        for name in params:
            if params[name].vary:
                params[name].correl = {}
                x0.append(params[name].value)
                x0name.append(params[name].name)
            has_expr = has_expr or params[name].expr is not None
        n = len(x0)

        # define epsilon
        if epsilon is None:
            EPS = np.MachAr().eps
            epsilon = EPS**(1./4)

        # maximum epsilon
        epsilon_max = 1000. * epsilon

        calchess = True
        while calchess:
            # allocate space for the hessian
            hessian = np.zeros((n, n))
            # the next loop fill the hessian matrix
            xx = x0
            if diff == 'forward':
                f0 = self.approx_fprime(optlocal, f, epsilon,
                                        neval = None, fcn_args=fcn_args)
            for j in range(n):
                d = epsilon * max(np.abs(x0[j]), 0.1)
                # forward difference
                params[x0name[j]].value = xx[j] + d
                params.update_constraints()
                ff = self.approx_fprime(optlocal, f, epsilon, neval=j,
                                        fcn_args=fcn_args)
                if diff == 'central':
                    # backward difference
                    params[x0name[j]].value = xx[j] - d
                    params.update_constraints()
                    fb = self.approx_fprime(optlocal, f, epsilon, neval=j,
                                            fcn_args=fcn_args)
                    hessian[:j+1, j] = (ff-fb) / (2*d)
                else:
                    hessian[:j+1, j] = (ff-f0[:j+1]) / d
                hessian[j, :j+1] = hessian[:j+1, j]
                # restore initial value of x0
                params[x0name[j]].value = xx[j]
                params.update_constraints()

            # check if hessian does not has NaN
            if not np.isnan(hessian).any():
                cov = np.linalg.pinv(hessian)
                if np.amin(np.diag(cov)) <= 0:
                    # find nearest positive semi-definite matrix
                    try:
                        cov = np.linalg.pinv(self.nearPSD(hessian))
                    except:
                        epsilon = epsilon * 10.
                        if epsilon > epsilon_max:
                            calchess = False
                if np.amin(np.diag(cov)) > 0:
                    corr,std = cov2corr(cov, return_std=True)
                    if np.isnan(std).any():
                        epsilon = epsilon * 10.
                        if epsilon > epsilon_max:
                            calchess = False
                    else:
                        calchess = False
                else:
                    epsilon = epsilon * 10.
                    if epsilon > epsilon_max:
                        calchess = False
                        corr,std = cov2corr(cov, return_std=True)
            else:
                # if hessian has NaN, increase epsilon and recalculate hessian
                epsilon = epsilon * 10.
                if epsilon > epsilon_max:
                    calchess = False
                    std = np.empty(n) * np.nan

        # fill optresult with std, covar and corr values
        if not np.isnan(std).any():
            # initialize covariance matrix
            optresult.covar = np.zeros((n,n))
            for j in range(n):
                if optresult.params[x0name[j]].vary:
                    covidj = optresult.var_names.index(x0name[j])
                    # restore initial parameter values in optresult
                    optresult.params[x0name[j]].init_value = initvalue[j]
                    # check if rmeas differs significantly (3*std) from zero.
                    if (cutoff and x0name[j] == 'res_r'
                        and 3*std[j] > optresult.params[x0name[j]].value):
                        # if not, then the assumption of normality
                        # would not be valid
                        # standard deviation and correlation
                        # will not be displayed
                        optresult.params[x0name[j]].stderr = None
                        optresult.params[x0name[j]].correl = {}
                        for k in range(0, j):
                            if optresult.params[x0name[k]].vary:
                                covidk = optresult.var_names.index(x0name[k])
                                (optresult.params[x0name[k]]
                                 .correl[x0name[j]]) = 0.
                                (optresult.params[x0name[j]]
                                 .correl[x0name[k]]) = 0.
                                optresult.covar[covidj,covidk] = None
                                optresult.covar[covidk,covidj] = None
                    else:
                        optresult.params[x0name[j]].stderr = std[j]
                        optresult.params[x0name[j]].correl = {}
                        for k in range(n):
                            if optresult.params[x0name[k]].vary:
                                covidk = optresult.var_names.index(x0name[k])
                                (optresult.params[x0name[j]]
                                 .correl[x0name[k]]) = corr[j,k]
                                optresult.covar[covidj,covidk] = cov[j,k]

        uvars = None
        if has_expr:
            # uncertainties on constrained parameters:
            #   get values with uncertainties (including correlations),
            #   temporarily set Parameter values to these,
            #   re-evaluate contrained parameters to extract stderr
            #   and then set Parameters back to best-fit value
            try:
                uvars = lmfit.uncertainties.correlated_values(x0,
                                                              optresult.covar)
            except (np.linalg.LinAlgError, ValueError, AttributeError):
                uvars = None
            if uvars is not None:
                for par in optresult.params.values():
                    lmfit.minimizer.eval_stderr(par, uvars,
                                                optresult.var_names,
                                                optresult.params)
                # restore nominal values
                for v, nam in zip(uvars, optresult.var_names):
                    optresult.params[nam].value = v.nominal_value

        return optresult

    @staticmethod
    def approx_fprime(optresult, f, epsilon, neval = None, fcn_args=(),
                      diff='forward'):
        """
        Finite-difference approximation of the gradient of a scalar function

        Parameters
        ----------
        optresult : Minimizer object
            output from :func:`lmfit.Minimize` containing optimization results
        f : callable
            The function of which to determine the gradient
            (partial derivatives).
            Should take `par` as first argument, other arguments to `f` can be
            supplied in ``*args``.  Should return a scalar, the value of the
            function at `par`.
        epsilon : array_like
            Increment to `par` to use for determining the function gradient.
            If a scalar, uses the same finite difference delta for all partial
            derivatives.  If an array, should contain one value per element of
            `par`.
        neval integer
            number of gradient evaluations
        args : list (optional)
            Any other arguments that are to be passed to `f`
        diff : string
            forward or central difference scheme (default: forward)

        Returns
        -------
        grad : ndarray
            The partial derivatives of `f` to `xk`.

        Notes
        -----
        The function gradient determined by the central finite difference
        is:

        .. math::
            \\partial\\mathcal{L}/\\partial\\psi_i
            =\\frac{\\mathcal{L}(\\psi+e_i)- \\mathcal{L}(\\psi-e_i)}{2e_i}

        The function gradient determined by the forward finite difference
        is:

        .. math::
            \\partial\\mathcal{L}/\\partial\\psi_i
            =\\frac{\\mathcal{L}(\\psi+e_i)- \\mathcal{L}(\\psi)}{e_i}

        """
        xk = []
        xkname = []
        for name in optresult.params:
            if optresult.params[name].vary:
                xk.append(optresult.params[name].value)
                xkname.append(optresult.params[name].name)

        if epsilon is None:
            EPS = np.MachAr().eps
            epsilon = EPS**(1./3)

        if neval is None:
            nk = len(xk)
        else:
            nk = neval+1

        grad = np.zeros((nk,), float)
        ei = np.zeros((len(xk),), float)
        if diff == 'forward':
            g0 = f(*((optresult.params,) + fcn_args))
        for k in range(nk):
            ei[k] = max(np.abs(xk[k]), 0.1)
            d = epsilon * ei
            # perturb with forward difference
            optresult.params[xkname[k]].value = xk[k] + d[k]
            # update parameters with expressions
            optresult.params.update_constraints()
            gf = f(*((optresult.params,) + fcn_args))
            if diff == 'central':
                # perturb with backward difference
                optresult.params[xkname[k]].value = xk[k] - d[k]
                # update parameters with expressions
                optresult.params.update_constraints()
                gb = f(*((optresult.params,) + fcn_args))
                grad[k] = (gf-gb) / (2*d[k])
            else:
                grad[k] = (gf-g0) / d[k]
            optresult.params[xkname[k]].value = xk[k]
            # update parameters with expressions
            optresult.params.update_constraints()
            ei[k] = 0.0

        return np.array(grad)

    @staticmethod
    def nearPSD(A, epsilon=0):
        """
        Get the nearest Positive Semi-Definite matrix of A

        Parameters
        ----------
        A : array
            square matrix
        epsilon : float
            cut-off value for eigenvalues (default=0)

        Returns
        -------
        out : array
           nearest Positive Semi-Definite matrix of A

        Notes
        -----
        The coefficient of determination is defined as:

        .. math::
            R^2 = \\frac{var(ts) - var(res)}{var(ts)}

        """
        n = A.shape[0]
        eigval, eigvec = np.linalg.eig(A)
        val = np.matrix(np.maximum(eigval,epsilon))
        vec = np.matrix(eigvec)
        with np.errstate(divide='ignore', invalid='ignore'):
            T = 1 / (np.multiply(vec,vec) * val.T)
            T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
            B = T  * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
        out = B * B.T

        return out
