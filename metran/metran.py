"""This module contains the Metran class in Pastas."""

from logging import getLogger
from os import getlogin

import numpy as np
from pandas import (DataFrame, DatetimeIndex, Series, Timedelta, Timestamp,
                    concat)
from pastas.timeseries import TimeSeries
from pastas.utils import initialize_logger, validate_name
from pastas.version import __version__

from .factoranalysis import FactorAnalysis
from .kalmanfilter import SPKalmanFilter
from .solver import ScipySolve

logger = getLogger(__name__)
initialize_logger(logger)


class Metran:
    """Class for the Pastas Metran model.

    Parameters
    ----------
    oseries: pandas.DataFrame, list of pandas.Series or pastas.TimeSeries
        Time series to be analyzed. Index must be DatetimeIndex.
        The series can be non-equidistant.
    name: str, optional
        String with the name of the model. The default is 'Cluster'
    tmin: str, optional
        String with a start date for the simulation period (E.g. '1980').
        If none is provided, the tmin from the oseries is used.
    tmax: str, optional
        String with an end date for the simulation period (E.g. '2010').
        If none is provided, the tmax from the oseries is used.

    Returns
    -------
    mt: metran.Metran
        Metran instance.
    """

    def __init__(self, oseries, name='Cluster', tmin=None, tmax=None):
        # Default settings
        self.settings = {
            "tmin": None,
            "tmax": None,
            "freq": "D",
            "min_pairs": 20,
            "solver": None,
            "warmup": 1
        }

        if tmin is not None:
            self.settings["tmin"] = tmin
        if tmax is not None:
            self.settings["tmax"] = tmax

        # Initialize and rework observations
        self.nfactors = 0
        self.set_observations(oseries)
        self.parameters = DataFrame(columns=['initial', 'pmin', 'pmax',
                                             'vary', 'name'])
        self.set_init_parameters()

        # initialize attributes for solving
        self.fit = None

        self.name = validate_name(name)

        # File Information
        self.file_info = self._get_file_info()

    @property
    def nparam(self):
        return self.parameters.index.size

    @property
    def nstate(self):
        return self.nseries + self.nfactors

    def standardize(self, oseries):
        """Mathod to standardize series by subtracting mean and dividing by
        standard deviation.

        Parameters
        ----------
        oseries : pandas.DataFrame
            series to be standardized

        Returns
        -------
        pandas.DataFrame
            standardized series
        """
        std = oseries.std()
        mean = oseries.mean()
        self.oseries_std = np.array(std.values)
        self.oseries_mean = np.array(mean.values)
        return (oseries - mean) / std

    def truncate(self, oseries):
        """Method to set start and end of series.

        If tmin and/or tmax have been defined in self.settings, use
        these dates to trucate series. Dates with only NaN are being removed.

        Parameters
        ----------
        oseries : pandas.DataFrame
            series to be tructated

        Returns
        -------
        pandas.DataFrame
            truncated series
        """
        if self.settings["tmin"] is None:
            tmin = oseries.index.min()
        else:
            tmin = self.settings["tmin"]
        if self.settings["tmax"] is None:
            tmax = oseries.index.max()
        else:
            tmax = self.settings["tmax"]
        oseries = oseries.loc[tmin:tmax]
        return oseries.dropna(how='all')

    def test_cross_section(self, oseries=None, min_pairs=None):
        """Method to test whether series have enough cross-sectional data.

        Default threshold value is defined by self.settings["min_pairs"].

        Parameters
        ----------
        oseries : pandas.DataFrame, optional
            Time series to be evaluated. The default is None.
        min_pairs : int, optional
            Minimum number of cross-sectional data for each series.
            Should be greater than 1. The default is None.

        Raises
        ------
        Exception
            If one of the series has less than min_pairs of cross-sectional
            data and exception is raised.

        Returns
        -------
        None.
        """
        if min_pairs is None:
            min_pairs = self.settings["min_pairs"]
        if min_pairs == 0:
            logger.warning("min_pairs must be greater than 0.")
        if oseries is None:
            oseries = self.oseries.copy()
        pairs = Series(index=oseries.columns)
        oseries["count"] = oseries.count(axis=1)
        for s in pairs.index:
            pairs[s] = oseries.dropna(subset=[s, ])["count"].count()
        oseries = oseries.drop(["count"], axis=1)
        if pairs.min() < max(min_pairs, 1):
            err = pairs[pairs < min_pairs].index.tolist()
            msg = "Number of cross-sectional data is less than " \
                + str(min_pairs) + " for series " \
                + (', ').join([str(e) for e in err])
            raise Exception(msg)

    def get_factors(self, oseries):
        """Method to get factor loadings based on factor analysis.

        This method also gets some relevant results from the
        factor analysis including the eigenvalues, specificity,
        communality and percentage explained by factors (fep).

        Parameters
        ----------
        oseries : pandas.DataFrame
            Series to be analyzed.

        Returns
        -------
        None.
        """
        fa = FactorAnalysis()
        self.factors = fa.solve(oseries)
        self.eigval = fa.eigval
        if self.factors is not None:
            self.nfactors = self.factors.shape[1]
            self.specificity = fa.get_specificity()
            self.communality = fa.get_communality()
            self.fep = fa.fep
        else:
            self.nfactors = 0

    def _init_kalmanfilter(self, oseries):
        """Internal method to initialize Kalmanfilter for sequential
        processing.

        Parameters
        ----------
        oseries : pandas.DataFrame
            Series being processed by the Kalmanfilter.

        Returns
        -------
        None.
        """
        self.kf = SPKalmanFilter()
        self.kf.set_observations(oseries)

    def _phi(self, alpha):
        """Internal method to calculate autoregressive model parameter based on
        parameter alpha.

        Parameters
        ----------
        alpha : float
            model parameter

        Returns
        -------
        float
            autoregressive model parameter
        """
        a = Timedelta(1, self.settings["freq"]) / Timedelta("1d")
        return np.exp(-a / alpha)

    def get_transition_matrix(self, p=None, initial=False):
        """Method to get transition matrix of the Metran dynamic factor model.

        Parameters
        ----------
        p : pandas.Series, optional
            Model parameters. The default is None.
        initial : bool, optional
            Determines whether to use initial (True)
            or optimal (False) parameters. The default is False.

        Returns
        -------
        transition_matrix : numpy.ndarray
            Transition matrix
        """
        if p is None:
            p = self.get_parameters(initial)
        transition_matrix = np.zeros((self.nstate, self.nstate),
                                     dtype=np.float64)
        for n in range(self.nseries):
            name = "sdf" + str(n + 1) + "_alpha"
            transition_matrix[n, n] = self._phi(p[name])
        for n in range(self.nfactors):
            name = "cdf" + str(n + 1) + "_alpha"
            transition_matrix[self.nseries + n,
                              self.nseries + n] = self._phi(p[name])
        return transition_matrix

    def get_transition_covariance(self, p=None, initial=False):
        """Method to get transition covariance matrix of the Metran dynamic
        factor model.

        Parameters
        ----------
        p : pandas.Series, optional
            Model parameters. The default is None.
        initial : bool, optional
            Determines whether to use initial (True)
            or optimal (False) parameters. The default is False.

        Returns
        -------
        transition_covariance : numpy.ndarray
            Transition covariance matrix
        """
        if p is None:
            p = self.get_parameters(initial)
        transition_covariance = np.eye(self.nstate, dtype=np.float64)
        for n in range(self.nseries):
            name = "sdf" + str(n + 1) + "_alpha"
            transition_covariance[n, n] = (1 - self._phi(p[name]) ** 2) \
                * self.specificity[n]
        for n in range(self.nfactors):
            name = "cdf" + str(n + 1) + "_alpha"
            transition_covariance[self.nseries + n, self.nseries + n] = (
                1 - self._phi(p[name]) ** 2
            )
        return transition_covariance

    def get_transition_variance(self, p=None, initial=False):
        """Method to extract diagonal of transition covariance matrix to get
        the transition variance vector.

        Parameters
        ----------
        p : pandas.Series, optional
            Model parameters. The default is None.
        initial : bool, optional
            Determines whether to use initial (True)
            or optimal (False) parameters. The default is False.

        Returns
        -------
        transition_variance : numpy.ndarray
            Transition variance vector
        """
        if p is None:
            p = self.get_parameters(initial)
        return np.diag(self.get_transition_covariance(p))

    def get_observation_matrix(self, p=None, initial=False):
        """Method to get observation matrix of the Metran dynamic factor model.

        Parameters
        ----------
        p : pandas.Series, optional
            Model parameters. The default is None.
        initial : bool, optional
            Determines whether to use initial (True)
            or optimal (False) parameters. The default is False.

        Returns
        -------
        observation_matrix : numpy.ndarray
            Observation matrix
        """
        if p is None:
            p = self.get_parameters(initial)
        observation_matrix = np.zeros((self.nseries, self.nstate),
                                      dtype=np.float64)
        observation_matrix[:, :self.nseries] = np.eye(self.nseries)
        for n in range(self.nseries):
            for k in range(self.nfactors):
                observation_matrix[n, self.nseries + k] = self.factors[n, k]
        return observation_matrix

    def get_observation_variance(self):
        """Method to get observation matrix.

        Currently the observation variance is zero by default.

        Returns
        -------
        observation_variance : numpy.ndarray
            Observation variance vector
        """
        (self.nseries, _) = self.factors.shape
        observation_variance = np.zeros(self.nseries, dtype=np.float64)
        return observation_variance

    def _get_matrices(self, p, initial=False):
        """Internal method to get all matrices required to define the Metran
        dynamic factor model.

        Parameters
        ----------
        p : pandas.Series
            Model parameters.
        initial : bool, optional
            Determines whether to use initial (True)
            or optimal (False) parameters. The default is False.

        Returns
        -------
        numpy.ndarray
            Transition matrix.
        numpy.ndarray
            Transition covariance matrix.
        numpy.ndarray
            Observation matrix.
        numpy.ndarray
            Observation variance vector.
        """
        return (self.get_transition_matrix(p, initial),
                self.get_transition_covariance(p, initial),
                self.get_observation_matrix(p, initial),
                self.get_observation_variance())

    def get_parameters(self, initial=False):
        """Method to get all parameters from the individual objects.

        Parameters
        ----------
        initial: bool, optional
            True to get initial parameters, False to get optimized parameters.
            If optimized parameters do not exist, return initial parameters.

        Returns
        -------
        parameters: pandas.Series
            initial or optimal parameters.
        """
        if not(initial) and "optimal" in self.parameters:
            parameters = self.parameters["optimal"]
        else:
            parameters = self.parameters["initial"]

        return parameters

    def set_init_parameters(self):
        """Method to initialize parameters to be optimized.

        Returns
        -------
        None.
        """
        pinit_alpha = 10
        for n in range(self.nfactors):
            self.parameters.loc["cdf" + str(n + 1) + "_alpha"] = (
                pinit_alpha, 1e-5, None, True, "cdf")
        for n in range(self.nseries):
            self.parameters.loc["sdf" + str(n + 1) + "_alpha"] = (
                pinit_alpha, 1e-5, None, True, "sdf")

    def mask_observations(self, mask):
        """Method to mask observations for processing with Kalman filter or
        smoother.

        This method does NOT change the oseries itself. It only
        masks (hides) observations while running the
        Kalman filter or smoother, so that these observations
        are not used for updating the Kalman filter/smoother.

        Parameters
        ----------
        mask : pandas.DataFrame
            DataFrame with shape of oseries containing 0 or False
            for observation to be kept and 1 or True for observation
            to be masked (hidden).

        Returns
        -------
        None.
        """
        if mask.shape != self.oseries.shape:
            logger.error("Dimensions of mask " + str(mask.shape)
                         + " do not equal dimensions of series "
                         + str(self.oseries.shape)
                         + ". Mask cannot be applied.")
        else:
            oseries = self.oseries.mask(mask.astype(bool))
            self.kf.init_states()
            self.kf.set_observations(oseries)
            self.kf.mask = True

    def unmask_observations(self):
        """Method to unmask observation and reset observations to self.oseries.

        Returns
        -------
        None.
        """
        oseries = self.oseries
        self.kf.init_states()
        self.kf.set_observations(oseries)
        self.kf.mask = False

    def set_observations(self, oseries):
        """Method to rework oseries to pandas.DataFrame for further use in
        Metran class.

        Parameters
        ----------
        oseries : pandas.DataFrame
        or list/tuple of pandas.Series/pandas.DataFrame/pastas.TimeSeries
            Time series to be analyzed.

        Raises
        ------
        Exception
            - if a DataFrame within a list/tuple has more than one column
            - if input type is not correct
            - if number of series is less than 2
            - if index of Series/DataFrame is not a DatetimeIndex

        Returns
        -------
        None.
        """
        # combine series to DataFrame
        if isinstance(oseries, (list, tuple)):
            _oseries = []
            _names = []
            if len(oseries) > 1:
                for os in oseries:
                    if isinstance(os, TimeSeries):
                        _oseries.append(os.series)
                        _names.append(os.name)
                    elif isinstance(os, (Series, DataFrame)):
                        if isinstance(os, DataFrame):
                            if os.shape[1] > 1:
                                raise Exception("One or more series have "
                                                + "DataFrame with multiple "
                                                + "columns")
                            os = os.squeeze()
                        _oseries.append(os)
                        _names.append(os.name)
                self.snames = _names
                oseries = concat(_oseries, axis=1)
            else:
                oseries = DataFrame()
        elif isinstance(oseries, DataFrame):
            self.snames = oseries.columns
        else:
            raise Exception("Input type should be either a "
                            "list, tuple, or pandas.DataFrame")

        if oseries.shape[1] < 2:
            raise Exception("Metran requires at least 2 series, "
                            "found " + str(oseries.shape[1]))

        oseries = self.truncate(oseries)
        if type(oseries.index) == DatetimeIndex:
            oseries = oseries.asfreq("D")
            self.nseries = oseries.shape[1]
            self.oseries = self.standardize(oseries)
            self.test_cross_section()
        else:
            raise Exception("Index of series must be DatetimeIndex")

    def get_observations(self):
        """Returns series as available in Metran class.

        Returns
        -------
        pandas.DataFrame
            Time series.
        """
        return self.oseries

    def get_mle(self, p):
        """Method to obtain maximum likelihood estimate based on Kalman filter.

        Parameters
        ----------
        p : pandas.Series
            Model parameters.

        Returns
        -------
        mle: float
            Maximum likelihood estimate.
        """
        p = Series(p, index=self.parameters.index)
        self.kf.set_matrices(*self._get_matrices(p))
        self.kf.run_filter()
        mle = self.kf.get_mle()
        return mle

    def get_state_means(self, p=None, method="smoother"):
        """Method to get filtered or smoothed state means.

        Parameters
        ----------
        p : pandas.Series
            Model parameters. The default is None.
        method : str, optional
            Use "filter" to obtain filtered states, and
            "smoother" to obtain smoothed states. The default is "smoother".

        Returns
        -------
        state_means : pandas.DataFrame
            Filtered or smoothed states. Column names refer to
            specific dynamic factors (sdf) and common dynamic factors (cdf)
        """
        self._run_kalman(method, p=p)
        columns = ["sdf" + str(i + 1) for i in range(self.nseries)]
        columns.extend(["cdf" + str(i + 1) for i in range(self.nfactors)])
        if method == "filter":
            means = self.kf.filtered_state_means
        else:
            means = self.kf.smoothed_state_means
        state_means = DataFrame(means, index=self.oseries.index,
                                columns=columns)
        return state_means

    def get_state_variances(self, p=None, method="smoother"):
        """Method to get filtered or smoothed state variances.

        Parameters
        ----------
        p : pandas.Series
            Model parameters. The default is None.
        method : str, optional
            Use "filter" to obtain filtered variances, and
            "smoother" to obtain smoothed variances.
            The default is "smoother".

        Returns
        -------
        state_variances : pandas.DataFrame
            Filtered or smoothed variances. Column names refer to
            specific dynamic factors (sdf) and common dynamic factors (cdf)
        """
        self._run_kalman(method, p=p)
        with np.errstate(invalid='ignore'):
            if method == "filter":
                cov = self.kf.filtered_state_covariances
            else:
                cov = self.kf.smoothed_state_covariances
            n_timesteps = cov.shape[0]
            var = np.vstack([np.diag(cov[i]) for i in range(n_timesteps)])
        columns = ["sdf" + str(i + 1) for i in range(self.nseries)]
        columns.extend(["cdf" + str(i + 1) for i in range(self.nfactors)])
        state_variances = DataFrame(var, index=self.oseries.index,
                                    columns=columns)
        return state_variances

    def get_state(self, i, p=None, ci=True, method="smoother"):
        """Method to get filtered or smoothed mean for specific state,
        optionally including 95% confidence interval.

        Parameters
        ----------
        i : int
            index of state vector to be obtained
        p : pandas.Series
            Model parameters. The default is None.
        ci : bool, optional
            If True, include confidence interval in DataFrame.
            The default is True.
        method : str, optional
            Use "filter" to obtain filtered variances, and
            "smoother" to obtain smoothed variances.
            The default is "smoother".

        Returns
        -------
        state : pandas.DataFrame
            ith filtered or smoothed state (mean),
            optionally with 'lower' and 'upper' as
            lower and upper bounds of 95% confidence interval.
        """
        state = None
        if i < 0 or i >= self.nstate:
            logger.error("Value of i must be >=0 and <" + self.nstate)
        else:
            state = self.get_state_means(p=p, method=method).iloc[:, i]
            if ci:
                variances = self.get_state_variances(
                    p=p, method=method).iloc[:, i]
                iv = 1.96 * np.sqrt(variances)
                state = concat([state, state - iv, state + iv], axis=1)
                state.columns = ['mean', 'lower', 'upper']
        return state

    def get_projected_means(self, p=None, standardized=False,
                            method="smoother"):
        """Method to calculate projected means, which are the filtered/smoothed
        estimates (means) for the observed series.

        Parameters
        ----------
        p : pandas.Series
            Model parameters. The default is None.
        standardized : bool, optional
            If True, obtain estimates for standardized series.
            If False, obtain estimates for unstandardized series.
            The default is False.
        method : str, optional
            Use "filter" to obtain filtered estimates, and
            "smoother" to obtain smoothed estimates.
            The default is "smoother".

        Returns
        -------
        projected_means : pandas.DataFrame
            Filtered or smoothed estimates for observed series.
        """
        self._run_kalman(method, p=p)
        if standardized:
            observation_matrix = self.observation_matrix
        else:
            observation_matrix = self.get_scaled_observation_matrix()
        (means, _) = \
            self.kf.get_projected(observation_matrix, method=method)
        projected_means = \
            DataFrame(means, index=self.oseries.index,
                      columns=self.oseries.columns)
        return projected_means

    def get_projected_variances(self, p=None, standardized=False,
                                method="smoother"):
        """Method to calculate projected variances, which are the
        filtered/smoothed variances for the observed series.

        Parameters
        ----------
        p : pandas.Series
            Model parameters. The default is None.
        standardized : bool, optional
            If True, obtain estimates for standardized series.
            If False, obtain estimates for unstandardized series.
            The default is False.
        method : str, optional
            Use "filter" to obtain filtered estimates, and
            "smoother" to obtain smoothed estimates.
            The default is "smoother".

        Returns
        -------
        projected_means : pandas.DataFrame
            Filtered or smoothed variances for observed series.
        """
        self._run_kalman(method, p=p)
        if standardized:
            observation_matrix = self.observation_matrix
        else:
            observation_matrix = self.get_scaled_observation_matrix()
        (_, variances) = \
            self.kf.get_projected(observation_matrix, method=method)
        projected_variances = \
            DataFrame(variances, index=self.oseries.index,
                      columns=self.oseries.columns)
        return projected_variances

    def get_projection(self, name, p=None, ci=True, standardized=False,
                       method="smoother"):
        """Method to calculate projected means for specific series, optionally
        including 95% confidence interval.

        Parameters
        ----------
        name : str
            name of series to be obtained
        p : pandas.Series
            Model parameters. The default is None.
        ci : bool, optional
            If True, include confidence interval in DataFrame.
            The default is True.
        standardized : bool, optional
            If True, obtain estimates for standardized series.
            If False, obtain estimates for unstandardized series.
            The default is False.
        method : str, optional
            Use "filter" to obtain filtered estimates, and
            "smoother" to obtain smoothed estimates.
            The default is "smoother".

        Returns
        -------
        proj : pandas.DataFrame
            filtered or smoothed estimate (mean) for series 'name',
            optionally with 'lower' and 'upper' as
            lower and upper bounds of 95% confidence interval.
        """
        proj = None
        means = self.get_projected_means(p=p, standardized=standardized,
                                         method=method)
        if name in means.columns:
            proj = means.loc[:, name]
            if ci:
                variances = \
                    self.get_projected_variances(p=p,
                                                 standardized=standardized,
                                                 method=method).loc[:, name]
                iv = 1.96 * np.sqrt(variances)
                proj = concat([proj, proj - iv, proj + iv], axis=1)
                proj.columns = ['mean', 'lower', 'upper']
        else:
            logger.error("Unknown name: " + name)
        return proj

    def decompose_projection(self, name, p=None, standardized=False,
                             method="smoother"):
        """Method to decompose filtered/smoothed estimate for observed series
        into specific dynamic component (sdf) and the sum of common dynamic
        components (cdf)

        Parameters
        ----------
        name : str
            name of series to be obtained.
        p : pandas.Series
            Model parameters. The default is None.
        standardized : bool, optional
            If True, obtain estimates for standardized series.
            If False, obtain estimates for unstandardized series.
            The default is False.
        method : str, optional
            Use "filter" to obtain filtered estimates, and
            "smoother" to obtain smoothed estimates.
            The default is "smoother".

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with specific and common dynamic component
            for series with name 'name'.
        """
        df = None
        self._run_kalman(method, p=p)
        if standardized:
            observation_matrix = self.observation_matrix
        else:
            observation_matrix = self.get_scaled_observation_matrix()
        (sdf_means, cdf_means) = \
            self.kf.decompose_projected(observation_matrix, method=method)
        if name in self.oseries.columns:
            sdf = DataFrame(sdf_means,
                            index=self.oseries.index,
                            columns=self.oseries.columns)
            cdf = DataFrame(cdf_means,
                            index=self.oseries.index,
                            columns=self.oseries.columns)
            df = concat([sdf.loc[:, name], cdf.loc[:, name]], axis=1)
            df.columns = ["sdf", "cdf"]
        else:
            logger.error("Unknown name: " + name)
        return df

    def get_scaled_observation_matrix(self):
        """Method scale observation matrix by standard deviations of oseries.

        Returns
        -------
        observation_matrix: numpy.ndarray
            scaled observation matrix
        """
        scale = self.oseries_std
        observation_matrix = self.get_observation_matrix()
        np.fill_diagonal(observation_matrix[:, :self.nseries], scale)
        for i in range(self.nfactors):
            observation_matrix[:, self.nseries + i] = \
                np.multiply(scale, observation_matrix[:, self.nseries + i])
        return observation_matrix

    def _run_kalman(self, method, p=None):
        """Internal method to (re)run Kalman filter or smoother.

        Parameters
        ----------
        method : str, optional
            Use "filter" to run Kalman filter, and
            "smoother" to run Kalman smoother. The default is "smoother".
        p : pandas.Series
            Model parameters. The default is None.

        Returns
        -------
        None.
        """
        if method == "filter":
            if p is not None:
                self.kf.set_matrices(*self._get_matrices(p))
                self.kf.run_filter()
            elif self.kf.filtered_state_means is None:
                self.kf.run_filter()
        else:
            if p is not None:
                self.kf.set_matrices(*self._get_matrices(p))
                self.kf.run_smoother()
            elif self.kf.smoothed_state_means is None:
                self.kf.run_smoother()

    def solve(self, solver=None, report=True, **kwargs):
        """Method to solve the time series model.

        Parameters
        ----------
        solver: metran.solver.BaseSolver class, optional
            Class used to solve the model. Options are: mt.ScipySolve
            (default) or mt.LmfitSolve. A class is needed, not an instance
            of the class!
        report: bool, optional
            Print reports to the screen after optimization finished. This
            can also be manually triggered after optimization by calling
            print(mt.fit_report()) or print(mt.metran_report())
            on the Metran instance.
        **kwargs: dict, optional
            All keyword arguments will be passed onto minimization method
            from the solver.

        Notes
        -----
        - The solver object including some results are stored as mt.fit.
          From here one can access the covariance (mt.fit.pcov) and
          correlation matrix (mt.fit.pcor).
        - The solver returns a number of results after optimization. These
          are stored in mt.fit.result and can be accessed from there.
        """

        # Perform factor analysis to get factors
        self.get_factors(self.oseries)
        # Initialize Kalmanfilter
        self._init_kalmanfilter(self.oseries)
        # Initialize parameters
        self.set_init_parameters()

        # Store the solve instance
        if solver is None:
            if self.fit is None:
                self.fit = ScipySolve(mt=self)
        elif not issubclass(solver, self.fit.__class__):
            self.fit = solver(mt=self)

        self.settings["solver"] = self.fit._name

        # Solve model
        success, optimal, stderr = self.fit.solve(**kwargs)

        self.parameters["optimal"] = optimal
        self.parameters["stderr"] = stderr

        if not success:
            logger.warning("Model parameters could not be estimated well.")

        if report:
            if isinstance(report, str):
                output = report
            else:
                output = "full"
            print("\n" + self.fit_report(output=output))
            print("\n" + self.metran_report())

    def _get_file_info(self):
        """Internal method to get the file information.

        Returns
        -------
        file_info: dict
            dictionary with file information.
        """
        # Check if file_info already exists
        if hasattr(self, "file_info"):
            file_info = self.file_info
        else:
            file_info = {"date_created": Timestamp.now()}

        file_info["date_modified"] = Timestamp.now()
        file_info["pastas_version"] = __version__

        try:
            file_info["owner"] = getlogin()
        except:
            file_info["owner"] = "Unknown"

        return file_info

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
        """
        model = {
            "tmin": str(self.settings["tmin"]),
            "tmax": str(self.settings["tmax"]),
            "freq": self.settings["freq"],
            "solver": self.settings["solver"]
        }

        fit = {
            "obj": "{:.2f}".format(self.fit.obj_func),
            "nfev": self.fit.nfev,
            "AIC": "{:.2f}".format(self.fit.aic),
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
                 "{line}\n".format(name=self.name[:14],
                                   string=string.format(
                                       "", fill=' ', align='>', width=w),
                                   line=string.format("", fill='=', align='>', width=width))

        basic = ""
        w = max(width - 45, 0)
        for (val1, val2), (val3, val4) in zip(model.items(), fit.items()):
            val4 = string.format(val4, fill=' ', align='>', width=w)
            basic += "{:<8} {:<16} {:<7} {:}\n".format(val1, val2, val3, val4)

        # Create the parameters block
        parameters = "\nParameters ({n_param} were optimized)\n{line}\n" \
                     "{parameters}".format(n_param=parameters.vary.sum(),
                                           line=string.format("", fill='=', align='>',
                                                              width=width),
                                           parameters=parameters)

        if output == "full":
            cor = {}
            pcor = self.fit.pcor
            for idx in pcor:
                for col in pcor:
                    if ((np.abs(pcor.loc[idx, col]) > 0.5) and (idx != col)
                            and ((col, idx) not in cor.keys())):
                        cor[(idx, col)] = pcor.loc[idx, col].round(2)

            cor = DataFrame(data=cor.values(), index=cor.keys(),
                            columns=["rho"])
            if cor.shape[0] > 0:
                cor = cor.to_string(header=False)
            else:
                cor = "None"
            correlations = "\n\nParameter correlations |rho| > 0.5\n{}" \
                           "\n{}".format(string.format("", fill='=',
                                                       align='>',
                                                       width=width), cor)

        else:
            correlations = ""

        report = "{header}{basic}{parameters}{correlations}".format(
            header=header, basic=basic, parameters=parameters,
            correlations=correlations)

        return report

    def metran_report(self, output="full"):
        """Method that reports on the metran model results.

        Parameters
        ----------
        output: str, optional
            If any other value than "full" is provided, the state
            correlations will be removed from the output.

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
        names = [("sdf" + str(i + 1)).ljust(idx_width)
                 for i in range(self.nseries)]
        names.extend([("cdf" + str(i + 1)).ljust(idx_width)
                      for i in range(self.nfactors)])
        transition = DataFrame(np.array([phi, q]).T,
                               index=names,
                               columns=["phi", "q"])

        # Create the observation parameters block
        gamma = self.factors
        names = ["gamma" + str(i + 1) for i in range(self.nfactors)]
        observation = DataFrame(gamma,
                                index=self.oseries.columns,
                                columns=names)
        observation.loc[:, "scale"] = self.oseries_std
        observation.loc[:, "mean"] = self.oseries_mean

        # Determine the width of the metran_report based on the parameters
        width = max(len(transition.__str__().split("\n")[1]),
                    len(observation.__str__().split("\n")[1]), 44)
        string = "{:{fill}{align}{width}}"

        # # Create the first header with results factor analysis
        w = max(width - 43, 0)
        header = "Metran report {name:<14}{string}Factor Analysis\n" \
            "{line}\n".format(name=self.name[:14],
                              string=string.format(
                                  "", fill=' ', align='>', width=w),
                              line=string.format("", fill='=', align='>',
                                                 width=width))

        factors = ""
        for (val1, val2), (val3, val4) in zip(model.items(), fit.items()):
            val4 = string.format(val4, fill=' ', align='>', width=w)
            factors += "{:<8} {:<19} {:<7} {:}\n".format(val1, val2,
                                                         val3, val4)

        # Create the transition block
        transition = "\nState parameters\n{line}\n" \
            "{transition}\n".format(line=string.format(
                "", fill='=', align='>', width=width),
                transition=transition)

        # Create the observation block
        observation = "\nObservation parameters\n{line}\n" \
                      "{observation}\n".format(line=string.format(
                          "", fill='=', align='>', width=width),
                          observation=observation)

        # Create the communality block
        communality = "\nCommunality\n{line}\n" \
                      "{communality}\n".format(line=string.format(
                          "", fill='=', align='>', width=width),
                          communality=communality)

        if output == "full":
            cor = {}
            pcor = self.get_state_means().corr()
            for idx in pcor:
                for col in pcor:
                    if ((np.abs(pcor.loc[idx, col]) > 0.5) and (idx != col)
                            and ((col, idx) not in cor.keys())):
                        cor[(idx, col)] = pcor.loc[idx, col].round(2)

            cor = DataFrame(data=cor.values(), index=cor.keys(),
                            columns=["rho"])
            if cor.shape[0] > 0:
                cor = cor.to_string(header=False)
            else:
                cor = "None"
            correlations = "\nState correlations |rho| > 0.5\n{}" \
                           "\n{}\n".format(string.format("", fill='=',
                                                         align='>',
                                                         width=width), cor)
        else:
            correlations = ""

        report = "{header}{factors}{communality}{transition}" \
            "{observation}{correlations}".format(header=header,
                                                 factors=factors,
                                                 communality=communality,
                                                 transition=transition,
                                                 observation=observation,
                                                 correlations=correlations)

        return report