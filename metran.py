"""This module contains the Metran class in Pastas."""

from logging import getLogger

import numpy as np
from pandas import DataFrame, concat
from pastas.decorators import set_parameter
from pastas.modelstats import Statistics
from pastas.plots import Plotting
from pastas.timeseries import TimeSeries
from pastas.utils import validate_name

from factoranalysis import FactorAnalysis
from kalmanfilter import SPKalmanFilter
from solver import LmfitSolve


class Metran:
    """Class for the Pastas Metran model.

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

    def __init__(self, oseries, name=None):  # , metadata=None):

        self.logger = getLogger(__name__)

        # Construct the different model components
        _oseries = []
        for os in oseries:
            if isinstance(os, TimeSeries):
                _oseries.append(os.series)
            else:
                _oseries.append(os)
        oseries = concat(_oseries, axis=1)
        oseries = self.truncate(oseries)
        oseries = oseries.asfreq("D")
        self.oseries = self.standardize(oseries)

        if name is None:
            name = 'Cluster'
        self.name = validate_name(name)

        self.parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])

        self.get_factoranalysis(self.oseries)
        self.init_kalmanfilter(self.oseries)
        self.set_init_parameters()

        # Default solve/simulation settings
        self.settings = {
            "tmin": None,
            "tmax": None,
            "freq": "D",
            "solver": None,
            "warmup": 1
        }

        # File Information
        # self.file_info = self._get_file_info()

        # initialize some attributes for solving and simulation
        self.sim_index = None
        self.oseries_calib = None
        self.interpolate_simulation = None
        self.fit = None

        # Load other modules
        self.stats = Statistics(self)
        self.plots = Plotting(self)
        self.plot = self.plots.plot  # because we are lazy

    @property
    def nparam(self):
        return self.parameters.index.size

    def standardize(self, oseries):
        std = oseries.std()
        self.stdfac = np.array(std.values)
        return (oseries - oseries.mean()) / std

    def truncate(self, oseries):
        return oseries.dropna(how='all')

    def get_factoranalysis(self, oseries):
        fa = FactorAnalysis(oseries)
        self.factors = fa.solve()
        self.nfactors = self.factors.shape[1]
        self.specificity = fa.get_specificity()
        self.cdf_variance = fa.get_eigval_weight()

    def init_kalmanfilter(self, oseries):
        self.kf = SPKalmanFilter()
        self.kf.initialize(oseries)

    def get_transition_matrix(self, p=None, initial=False):
        if p is None:
            p = self.get_parameters(initial)
        (nsdf, ncdf) = self.factors.shape
        nstate = nsdf + ncdf
        transition_matrix = np.zeros((nstate, nstate))
        for n in range(nsdf):
            name = "sdf" + str(n + 1) + "_alpha"
            transition_matrix[n, n] = 1. - np.exp(-p[name])
        for n in range(ncdf):
            name = "cdf" + str(n + 1) + "_alpha"
            transition_matrix[nsdf + n, nsdf + n] = (
                1. - np.exp(-1 * p[name]))
        return transition_matrix

    def get_transition_covariance(self, p=None, initial=False):
        if p is None:
            p = self.get_parameters(initial)
        (nsdf, ncdf) = self.factors.shape
        nstate = nsdf + ncdf
        transition_covariance = np.eye(nstate)
        for n in range(nsdf):
            name = "sdf" + str(n + 1) + "_alpha"
            transition_covariance[n, n] = (
                1 - (
                    1. - np.exp(-1 * p[name]))**2) * self.specificity[n]
        for n in range(ncdf):
            name = "cdf" + str(n+1) + "_q"
            transition_covariance[nsdf+n, nsdf+n] =p[name]
            # name = "cdf" + str(n+1) + "_alpha"
            # transition_covariance[nsdf+n, nsdf+n] = (
            #     1 - (
            #     1. - np.exp(-1 * p[name]))**2)
        return transition_covariance

    def get_transition_variance(self, p=None, initial=False):
        if p is None:
            p = self.get_parameters(initial)
        return np.diag(self.get_transition_covariance(p))

    def get_observation_matrix(self, p=None, initial=False):
        if p is None:
            p = self.get_parameters(initial)
        (nsdf, ncdf) = self.factors.shape
        nstate = nsdf + ncdf
        observation_matrix = np.zeros((nsdf, nstate))
        observation_matrix[:, :nsdf] = np.eye(nsdf)
        for n in range(nsdf):
            for k in range(ncdf):
                name = "cdf" + str(k + 1) + "_c" + str(n + 1)
                observation_matrix[n, nsdf + k] = p[name]
        return observation_matrix

    def get_observation_variance(self):
        (nsdf, _) = self.factors.shape
        observation_variance = np.zeros(nsdf)
        return observation_variance

    def get_matrices(self, p):
        return (self.get_transition_matrix(p),
                self.get_transition_covariance(p),
                self.get_observation_matrix(p),
                self.get_observation_variance()
                )

    def set_init_parameters(self):
        pinit_alpha = 3
        pinit_q = 0.1
        (nsdf, ncdf) = self.factors.shape

        for n in range(ncdf):
            self.parameters.loc["cdf" + str(n + 1) + "_alpha"] = (
                pinit_alpha, 0, 10, True, "cdf")

        for n in range(ncdf):
            # if n == 0: # and parmulti['r'][n] == 0:
            #     # fix parameter to 1 for concentrated likelihood optimization
            #     # only valid if no measurement error R is included
            #     self.parameters.loc["cdf" + str(n+1) + "_q"] = (
            #         pinit_q, 0, None, False, "cdf")
            # else:
            self.parameters.loc["cdf" + str(n+1) + "_q"] = (
                pinit_q, 0, None, True, "cdf")

        for n in range(nsdf):
            self.parameters.loc["sdf" + str(n + 1) + "_alpha"] = (
                pinit_alpha, 0, 10, True, "sdf")

        # for n in range(nsdf):
        #     self.parameters.loc["sdf" + str(n+1) + "_q"] = (
        #         pinit_q, 0, None, True, "sdf")

        for n in range(nsdf):
            for k in range(ncdf):
                self.parameters.loc["cdf" + str(k + 1) + "_c" + str(n + 1)] = (
                    self.factors[n, k], None, None, False, "cdf")

    @set_parameter
    def _set_initial(self, name, value):
        """Internal method to set the initial parameter value.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "initial"] = value

    @set_parameter
    def _set_pmin(self, name, value):
        """Internal method to set the minimum value of the noisemodel.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "pmin"] = value

    @set_parameter
    def _set_pmax(self, name, value):
        """Internal method to set the maximum parameter values.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "pmax"] = value

    @set_parameter
    def _set_vary(self, name, value):
        """Internal method to set if the parameter is varied.

        Notes
        -----
        The preferred method for parameter setting is through the model.
        """
        self.parameters.loc[name, "vary"] = value

    def simulate(self, p):
        """Simulate.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters. Here, Alpha parameter used by the noisemodel.

        Returns
        -------
        SPKalmanFilter Class
            Results of Kalmanfilter
        """
        self.kf.set_matrices(*self.get_matrices(p))
        self.kf.runf95()

        return self.kf

    def get_parameters(self, initial=True):
        """Method to get all parameters from the individual objects.

        Parameters
        ----------
        initial: bool, optional
            True to get initial parameters, False to get optimized parameters.
            If optimized parameters do not exist, return initial parameters.

        Returns
        -------
        parameters: pandas.DataFrame
            pandas.Dataframe with the parameters.
        """
        if not(initial) and "optimal" in self.parameters:
            parameters = self.parameters["optimal"]
        else:
            parameters = self.parameters["initial"]

        return parameters

    @staticmethod
    def _scale_parameter(p, fact):
        p.value = fact * p.value
        p.init_value = fact * p.init_value
        if p.stderr is not None:
            p.stderr = fact * p.stderr
        return p

    def scale_covariances(self, params, fact):
        """Rescale optimized parameters and associated standard deviations.

        Parameters
        ----------
        result : class Parameters(OrderedDict)
            parameter results
        fact : float
            scaling factor

        Returns
        -------
        result : class Parameters(OrderedDict)
            scaled parameter results
        """

        for name in params:
            if name.endswith('_q'):
                params[name] = self._scale_parameter(
                    params[name], fact)

        return params

    def scale_parameters(self, params):
        dim = self.factors.shape[0]
        for n in range(dim):
            name = 'sdf' + str(n + 1) + '_q'
            params[name] = self._scale_parameter(params[name],
                                                 self.stdfac[n]**2)
            name = 'r' + str(n + 1)
            params[name] = self._scale_parameter(params[name],
                                                 self.stdfac[n]**2)
            for k in range(self.nfactors):
                name = 'cdf' + str(k + 1) + '_c' + str(n + 1)
                params[name] = self._scale_parameter(params[name],
                                                     self.stdfac[n])
        return params

    def solve(self, solver=None, report=True, **kwargs):
        """Method to solve the time series model.

        Parameters
        ----------
        solver: metran.solver.BaseSolver class, optional
            Class used to solve the model. Currently only ps.LmfitSolve
            is available. A class is needed, not an instance
            of the class!
        report: bool, optional
            Print a report to the screen after optimization finished. This
            can also be manually triggered after optimization by calling
            print(ml.fit_report()) on the Pastas model instance.
        **kwargs: dict, optional
            All keyword arguments will be passed onto minimization method
            from the solver. It depends on the solver used which arguments
            can be used.

        Notes
        -----
        - The solver object including some results are stored as ml.fit.
          From here one can access the covariance (ml.fit.pcov) and
          correlation matrix (ml.fit.pcor).
        - Each solver return a number of results after optimization. These
          solver specific results are stored in ml.fit.result and can be
          accessed from there.

        See Also
        --------
        pastas.solver
            Different solver objects are available to estimate parameters.
        """

        # Store the solve instance
        if solver is None:
            if self.fit is None:
                self.fit = LmfitSolve(ml=self)
        elif not issubclass(solver, self.fit.__class__):
            self.fit = solver(ml=self)

        self.settings["solver"] = self.fit._name

        # Solve model
        success, self.params = self.fit.solve(**kwargs)
        if not success:
            self.logger.warning("Model parameters could not be estimated "
                                "well.")

        self.parameters["optimal"] = np.array([p.value
                                               for p in self.params.values()])
        self.parameters["stderr"] = np.array([p.stderr
                                              for p in self.params.values()])

        if report:
            if isinstance(report, str):
                output = report
            else:
                output = "full"
            print(self.fit_report(output=output))

    def fit_report(self, output="full"):
        """Method that reports on the fit after a model is optimized.

        Parameters
        ----------
        output: str, optional
            If any other value than "full" is provided, the parameter
            correlations will be removed from the output.

        Returns
        -------
        report: str
            String with the report.

        Examples
        --------
        This method is called by the solve method if report=True, but can
        also be called on its own::

        >>> print(ml.fit_report)

        Notes
        -----
        The reported values for the fit use the residuals time series where
        possible. If interpolation is used this means that the result may
        slightly differ compared to using ml.simulate() and ml.observations().
        """
        model = {
            "nfev": self.fit.nfev,
            # "nobs": self.observations().index.size,
            # "noise": str(self.settings["noise"]),
            "tmin": str(self.settings["tmin"]),
            "tmax": str(self.settings["tmax"]),
            "freq": self.settings["freq"],
            "warmup": str(self.settings["warmup"]),
            "solver": self.settings["solver"]
        }

        fit = {
            "Obj": "{:.2f}".format(self.fit.obj_func),
        }

        parameters = self.parameters.loc[:, ["optimal", "stderr",
                                             "initial", "vary"]]
        stderr = parameters.loc[:, "stderr"] / parameters.loc[:, "optimal"]
        parameters.loc[:, "stderr"] = "-"
        parameters.loc[parameters["vary"],
                       "stderr"] = stderr.abs().apply("\u00B1{:.2%}".format)
        parameters.loc[~parameters["vary"].astype(bool), "initial"] = "-"

        # Determine the width of the fit_report based on the parameters
        width = len(parameters.__str__().split("\n")[1])
        string = "{:{fill}{align}{width}}"

        # Create the first header with model information and stats
        w = max(width - 44, 0)
        header = "Fit report {name:<16}{string}Fit Statistics\n" \
                 "{line}\n".format(
                     name=self.name[:14],
                     string=string.format("", fill=' ', align='>', width=w),
                     line=string.format("", fill='=', align='>', width=width))

        basic = ""
        w = max(width - 45, 0)
        for (val1, val2), (val3, val4) in zip(model.items(), fit.items()):
            val4 = string.format(val4, fill=' ', align='>', width=w)
            basic += "{:<8} {:<22} {:<12} {:}\n".format(val1, val2, val3, val4)

        # Create the parameters block
        parameters = "\nParameters ({n_param} were optimized)\n{line}\n" \
                     "{parameters}".format(
                         n_param=parameters.vary.sum(),
                         line=string.format(
                             "", fill='=', align='>', width=width),
                         parameters=parameters)

        if output == "full":
            cor = {}
            pcor = self.fit.pcor
            for idx in pcor:
                for col in pcor:
                    if (np.abs(pcor.loc[idx, col]) > 0.5) and (idx != col) \
                            and ((col, idx) not in cor.keys()):
                        cor[(idx, col)] = pcor.loc[idx, col].round(2)

            cor = DataFrame(data=cor.values(), index=cor.keys(),
                            columns=["rho"])
            correlations = "\n\nParameter correlations |rho| > 0.5\n{}" \
                           "\n{}".format(string.format("", fill='=', align='>',
                                                       width=width),
                                         cor.to_string(header=False))
        else:
            correlations = ""

        report = "{header}{basic}{parameters}{correlations}".format(
            header=header, basic=basic, parameters=parameters,
            correlations=correlations)

        return report
