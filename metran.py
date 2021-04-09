"""This module contains the Metran class in Pastas."""

from logging import getLogger

import numpy as np
from pandas import DataFrame, Series, Timedelta, concat
from pastas.decorators import set_parameter
from pastas.timeseries import TimeSeries
from pastas.utils import validate_name

from factoranalysis import FactorAnalysis
from kalmanfilter import SPKalmanFilter
from solver import LmfitSolve

logger = getLogger(__name__)

class Metran:
    """Class for the Pastas Metran model.

    Parameters
    ----------
    oseries: pandas.DataFrame, list of pandas.Series or pastas.TimeSeries
        Time series to be analyzed. The series can be non-equidistant.
    name: str, optional
        String with the name of the model, used in plotting and saving.

    Returns
    -------
    mt: pastas.metran.Metran
        Pastas Metran instance.

    Examples
    --------
    A minimal working example of the Metran class is shown below:

    >>> oseries = pd.Series([1,2,1], index=pd.to_datetime(range(3), unit="D"))
    >>> mt = Metran(oseries)
    """

    def __init__(self, oseries, name=None):

        # Default solve/simulation settings
        self.settings = {
            "tmin": None,
            "tmax": None,
            "freq": "D",
            "solver": None,
            "warmup": 1
        }

        if isinstance(oseries, (list, tuple)):
            _oseries = []
            _names = []
            for os in oseries:
                if isinstance(os, TimeSeries):
                    _oseries.append(os.series)
                    _names.append(os.name)
                elif isinstance(os, (Series, DataFrame)):
                    if isinstance(os, DataFrame):
                        if os.shape[1] > 1:
                            logger.error("One or more oseries have DataFrame "
                                         "with multiple columns")
                        os = os.squeeze()
                    _oseries.append(os)
                    _names.append(os.name)
            self.snames = _names
            oseries = concat(_oseries, axis=1)
        elif isinstance(oseries, DataFrame):
            self.snames = oseries.columns
        else:
            logger.error("Input type should be either a "
                         "list, tuple, or pandas.DataFrame")

        oseries = self.truncate(oseries)
        oseries = oseries.asfreq("D")
        self.nseries = oseries.shape[1]
        self.oseries = self.standardize(oseries)

        if name is None:
            name = 'Cluster'
        self.name = validate_name(name)

        self.parameters = DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])

        self.get_factoranalysis(self.oseries)
        self.init_kalmanfilter(self.oseries)
        self.nstate = self.nseries + self.nfactors

        self.set_init_parameters()

        # File Information
        # self.file_info = self._get_file_info()

        # initialize some attributes for solving and simulation
        self.sim_index = None
        self.oseries_calib = None
        self.interpolate_simulation = None
        self.fit = None


    @property
    def nparam(self):
        return self.parameters.index.size

    def standardize(self, oseries):
        std = oseries.std()
        mean = oseries.mean()
        self.oseries_std = np.array(std.values)
        self.oseries_mean = np.array(mean.values)
        return (oseries - mean) / std

    def truncate(self, oseries):
        return oseries.dropna(how='all')

    def get_factoranalysis(self, oseries):
        fa = FactorAnalysis(oseries)
        self.factors = fa.solve()
        self.eigval = fa.eigval
        self.nfactors = self.factors.shape[1]
        self.specificity = fa.get_specificity()
        self.communality = fa.get_communality()
        self.fep = 100 * np.sum(fa.get_eigval_weight()[:self.nfactors])

    def init_kalmanfilter(self, oseries):
        self.kf = SPKalmanFilter()
        self.kf.initialize(oseries)

    def _phi(self, p):
        a = Timedelta(1, self.settings["freq"]) / Timedelta("1d")
        return np.exp(-a/p)

    def _dalpha(self, p):
        a = Timedelta(1, self.settings["freq"]) / Timedelta("1d")
        return np.exp(-a/p) * a / p**2

    def get_transition_matrix(self, p=None, initial=False):
        if p is None:
            p = self.get_parameters(initial)
        transition_matrix = np.zeros((self.nstate, self.nstate),
                                     dtype=np.float64)
        for n in range(self.nseries):
            name = "sdf" + str(n+1) + "_alpha"
            transition_matrix[n, n] = self._phi(p[name])
        for n in range(self.nfactors):
            name = "cdf" + str(n+1) + "_alpha"
            transition_matrix[self.nseries + n,
                              self.nseries + n] = self._phi(p[name])
        return transition_matrix

    def get_transition_covariance(self, p=None, initial=False):
        if p is None:
            p = self.get_parameters(initial)
        transition_covariance = np.eye(self.nstate, dtype=np.float64)
        for n in range(self.nseries):
            name = "sdf" + str(n+1) + "_alpha"
            transition_covariance[n, n] = (
                1 - self._phi(p[name])**2) * self.specificity[n]
        for n in range(self.nfactors):
            name = "cdf" + str(n+1) + "_alpha"
            transition_covariance[self.nseries+n,
                                  self.nseries+n] = (1 - self._phi(p[name])**2)
        return transition_covariance

    def get_transition_variance(self, p=None, initial=False):
        if p is None:
            p = self.get_parameters(initial)
        return np.diag(self.get_transition_covariance(p))

    def get_observation_matrix(self, p=None, initial=False):
        if p is None:
            p = self.get_parameters(initial)
        observation_matrix = np.zeros((self.nseries, self.nstate),
                                      dtype=np.float64)
        observation_matrix[:, :self.nseries] = np.eye(self.nseries)
        for n in range(self.nseries):
            for k in range(self.nfactors):
                observation_matrix[n, self.nseries+k] = self.factors[n, k]
        return observation_matrix

    def get_observation_variance(self):
        (self.nseries, _) = self.factors.shape
        observation_variance = np.zeros(self.nseries, dtype=np.float64)
        return observation_variance

    def get_matrices(self, p):
        return (self.get_transition_matrix(p),
                self.get_transition_covariance(p),
                self.get_observation_matrix(p),
                self.get_observation_variance()
                )

    def set_init_parameters(self):
        pinit_alpha = 10
        for n in range(self.nfactors):
            self.parameters.loc["cdf" + str(n + 1) + "_alpha"] = (
                pinit_alpha, 1e-5, None, True, "cdf")
        for n in range(self.nseries):
            self.parameters.loc["sdf" + str(n + 1) + "_alpha"] = (
                pinit_alpha, None, None, True, "sdf")

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

    def get_mle(self, p):
        """Run Kalmanfilter and calculate maximum likelihood estimate.

        Parameters
        ----------
        p: array_like
            array_like object with the values as floats representing the
            model parameters.

        Returns
        -------
        mle: float
            maximum likelihood estimate
        """
        self.kf.set_matrices(*self.get_matrices(p))
        self.kf.run_filter()
        mle = self.kf.get_mle()
        return mle

    def get_state_means(self, p=None):
        self._run_smoother(p)
        columns = ["state" + str(i) for i in range(self.nstate)]
        state_means = DataFrame(self.kf.smoothed_state_means,
                                index=self.oseries.index,
                                columns=columns)
        state_means.name = "state_means"
        return state_means

    def get_state_variances(self, p=None):
        self._run_smoother(p)
        with np.errstate(invalid='ignore'):
            n_timesteps = self.kf.smoothed_state_covariances.shape[0]
            var = np.vstack([np.diag(self.kf.smoothed_state_covariances[i])
                            for i in range(n_timesteps)])
        columns = ["state" + str(i) for i in range(self.nstate)]
        state_variances = DataFrame(var, index=self.oseries.index,
                                    columns=columns)
        state_variances.name = "state_variances"
        return state_variances

    def get_projected_means(self, p=None, standardized=False):
        self._run_smoother(p)
        if standardized:
            observation_matrix = self.observation_matrix
        else:
            observation_matrix = self.get_scaled_observation_matrix()
        (smoothed_projected_means, _) = \
            self.kf.get_projected(observation_matrix, method="smoothed")
        projected_means = \
            DataFrame(smoothed_projected_means,
                      index=self.oseries.index,
                      columns=self.oseries.columns)
        projected_means.name="projected_means"
        return projected_means

    def get_projected_variances(self, p=None, standardized=False):
        self._run_smoother(p)
        if standardized:
            observation_matrix = self.observation_matrix
        else:
            observation_matrix = self.get_scaled_observation_matrix()
        (_, smoothed_projected_variances) = \
            self.kf.get_projected(observation_matrix, method="smoothed")
        projected_variances = \
            DataFrame(smoothed_projected_variances,
                      index=self.oseries.index,
                      columns=self.oseries.columns)
        projected_variances.name="projected_variances"
        return projected_variances

    def get_state(self, i, ci=True, p=None):
        self._run_smoother(p)
        if i < 0 or i >= self.nstate:
            self.logger.error("Value of i must be >=0 and <" + self.nstate)
        df = self.get_state_means(p).iloc[:, i]
        if ci:
            variances = self.get_state_variances(p).iloc[:, i]
            iv = 1.96 * np.sqrt(variances)
            df = concat([df, df - iv, df + iv], axis=1)
            df.columns = ['mean', 'lower', 'upper']
        return df

    def get_projection(self, name, ci=True, p=None):
        self._run_smoother(p)
        means = self.get_projected_means(p)
        if name not in means.columns:
            self.logger.error("Unknown name: " + name)
        means = means.loc[:, name]
        if ci:
            variances = self.get_projected_variances(p).loc[:, name]
            iv = 1.96 * np.sqrt(variances)
            df = concat([means, means - iv, means + iv], axis=1)
            df.columns = ['mean', 'lower', 'upper']
        return df

    def _run_smoother(self, p=None):
        if p is not None:
            self.kf.set_matrices(*self.get_matrices(p))
            self.kf.run_smoother()
        elif self.kf.smoothed_state_means is None:
            self.kf.run_smoother()

    def _run_filter(self, p=None):
        if p is not None:
            self.kf.set_matrices(*self.get_matrices(p))
            self.kf.run_filter()

    def get_filtered_state_means(self, p=None):
        """Get filtered state mean (expected value)
           as calculated by Kalmanfilter.

        Parameters
        ----------
        p: array_like, optional
            array_like object with the values as floats representing the
            model parameters.

        Returns
        -------
        filtered_state_means: pandas.DataFrame
            columns with filtered state mean for each time step
        """
        self._run_filter(p)
        columns = ["state" + str(i) for i in range(self.nstate)]
        filtered_state_means = DataFrame(index=self.oseries.index,
                                         data=self.kf.filtered_state_means,
                                         columns=columns)
        return filtered_state_means

    def get_filtered_state_covariances(self, p=None):
        """Get filtered state covariance as calculated by Kalmanfilter.

        Parameters
        ----------
        p: array_like, optional
            array_like object with the values as floats representing the
            model parameters.

        Returns
        -------
        filtered_state_covariances: numpy.ndarray (3 dimensional)
            filtered state covariance matrix for each time step
        """
        self._run_filter(p)
        return self.kf.filtered_state_covariances

    def get_filtered_state_variances(self, p=None):
        """Get filtered state variance as calculated by Kalmanfilter.

        Parameters
        ----------
        p: array_like, optional
            array_like object with the values as floats representing the
            model parameters.

        Returns
        -------
        filtered_state_variances: pandas.DataFrame
            columns with filtered state variance for each time step
        """
        self._run_filter(p)
        variances = np.zeros(self.kf.filtered_state_means)
        for t in range(len(self.filtered_state_means)):
            variances[t, :] = np.diag(self.filtered_state_covariances)
        columns = ["state" + str(i) for i in range(self.nstate)]
        filtered_state_variances = DataFrame(index=self.oseries.index,
                                         data=variances,
                                         columns=columns)
        return filtered_state_variances

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

    def get_scaled_observation_matrix(self):
        """Method scale observation matrix by standard deviations of oseries.

        Returns
        -------
        observation_matrix: numpy.ndarray
            scaled observation matrix
        """
        scale = self.oseries_std
        observation_matrix = self.get_observation_matrix()
        np.fill_diagonal(observation_matrix[:,:self.nseries], scale)
        for i in range(self.nfactors):
            observation_matrix[:, self.nseries + i] = \
                np.multiply(scale, observation_matrix[:, self.nseries + i])
        return observation_matrix

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
            print(mt.fit_report()) on the Pastas model instance.
        **kwargs: dict, optional
            All keyword arguments will be passed onto minimization method
            from the solver. It depends on the solver used which arguments
            can be used.

        Notes
        -----
        - The solver object including some results are stored as mt.fit.
          From here one can access the covariance (mt.fit.pcov) and
          correlation matrix (mt.fit.pcor).
        - Each solver return a number of results after optimization. These
          solver specific results are stored in mt.fit.result and can be
          accessed from there.

        See Also
        --------
        pastas.solver
            Different solver objects are available to estimate parameters.
        """

        # Store the solve instance
        if solver is None:
            if self.fit is None:
                self.fit = LmfitSolve(mt=self)
        elif not issubclass(solver, self.fit.__class__):
            self.fit = solver(mt=self)

        self.settings["solver"] = self.fit._name

        # Solve model
        success, self.params = self.fit.solve(**kwargs)
        if not success:
            logger.warning("Model parameters could not be estimated "
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
            print(self.fit_report(output=output) + "\n")
            print(self.metran_report())

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

        >>> print(mt.fit_report())

        Notes
        -----
        The reported values for the fit use the residuals time series where
        possible. If interpolation is used this means that the result may
        slightly differ compared to using mt.simulate() and mt.observations().
        """
        model = {
            "tmin": str(self.settings["tmin"]),
            "tmax": str(self.settings["tmax"]),
            "freq": self.settings["freq"],
            "warmup": str(self.settings["warmup"]),
            "solver": self.settings["solver"]
        }

        fit = {
            "Obj": "{:.2f}".format(self.fit.obj_func),
            "nfev": self.fit.nfev,
            "AIC": "{:.2f}".format(self.fit.aic),
            "BIC": "{:.2f}".format(self.fit.bic),
            "": ""
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
            basic += "{:<8} {:<16} {:<7} {:}\n".format(val1, val2, val3, val4)

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

    def metran_report(self):
        """Method that reports on the metran model results.

        Returns
        -------
        report: str
            String with the report.

        Examples
        --------
        This method is called by the solve method if report=True, but can
        also be called on its own::

        >>> print(mt.metran_report())
        """

        model = {
            "tmin": str(self.settings["tmin"]),
            "tmax": str(self.settings["tmax"]),
            "freq": self.settings["freq"]
        }

        fit = {
            "nfct": str(self.nfactors),
            "fep": "{:.2f}%".format(self.fep),
            "": ""
        }

        # Create the communality block
        communality = Series(self.communality,
                             index=self.oseries.columns,
                             name="")
        communality = communality.apply("{:.2%}".format).to_frame()

        # get width of index to align state parameters index
        idx_width = int(max([len(n) for n in communality.index]))

        # Create the state parameters block
        phi = np.diag(self.get_transition_matrix())
        q = self.get_transition_variance()
        names = [("sdf" + str(i+1)).ljust(idx_width)
                 for i in range(self.nseries)]
        names.extend([("cdf" + str(i+1)).ljust(idx_width)
                      for i in range(self.nfactors)])
        transition = DataFrame(np.array([phi, q]).T,
                               index=names,
                               columns=["phi", "q"])

        # Create the observation parameters block
        gamma = self.factors
        names = ["gamma" + str(i+1) for i in range(self.nfactors)]
        observation = DataFrame(gamma,
                                index=self.oseries.columns,
                                columns=names)
        observation.loc[:, "scale"] = self.oseries_std
        observation.loc[:, "mean"] = self.oseries_mean

        # Determine the width of the fit_report based on the parameters
        width = max(len(transition.__str__().split("\n")[1]),
                    len(observation.__str__().split("\n")[1]), 44)
        string = "{:{fill}{align}{width}}"

        # # Create the first header with results factor analysis
        w = max(width - 43, 0)
        header = "Metran report {name:<14}{string}Factor Analysis\n" \
                  "{line}\n".format(
                      name=self.name[:14],
                      string=string.format("", fill=' ', align='>', width=w),
                      line=string.format("", fill='=', align='>', width=width))

        factors = ""
        for (val1, val2), (val3, val4) in zip(model.items(), fit.items()):
            val4 = string.format(val4, fill=' ', align='>', width=w)
            factors += "{:<8} {:<19} {:<7} {:}\n".format(val1, val2,
                                                         val3, val4)

        # Create the transition block
        transition = "\nState parameters\n{line}\n" \
                      "{transition}\n".format(
                          line=string.format(
                              "", fill='=', align='>', width=width),
                          transition=transition)

        # Create the observation block
        observation = "\nObservation parameters\n{line}\n" \
                      "{observation}\n".format(
                          line=string.format(
                              "", fill='=', align='>', width=width),
                          observation=observation)

        # Create the communality block
        communality = "\nCommunality\n{line}\n" \
                      "{communality}\n".format(
                          line=string.format(
                              "", fill='=', align='>', width=width),
                          communality=communality)

        report = "{header}{factors}{communality}{transition}" \
            "{observation}".format(
            header=header, factors=factors, communality=communality,
            transition=transition, observation=observation)

        return report
