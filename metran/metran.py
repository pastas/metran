"""The Metran model class."""

from logging import getLogger
from os import getlogin

import numpy as np
from pandas import (DataFrame, DatetimeIndex, Series, Timedelta, Timestamp,
                    concat)
from pandas.tseries.frequencies import to_offset
from pastas.timeseries import TimeSeries
from pastas.utils import (frequency_is_supported, initialize_logger,
                          validate_name)
from pastas.version import __version__
from scipy.stats import norm

from .factoranalysis import FactorAnalysis
from .kalmanfilter import SPKalmanFilter
from .plots import MetranPlot
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
    freq: str, optional
        String with the frequency the stressmodels are simulated. Must
        be one of the following (D, h, m, s, ms, us, ns) or a multiple of
        that e.g. "7D".
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

    def __init__(self, oseries, name='Cluster', freq=None,
                 tmin=None, tmax=None):
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
        if freq is not None:
            self.settings["freq"] = frequency_is_supported(freq)

        # Initialize and rework observations
        self.nfactors = 0
        self.set_observations(oseries)
        self.parameters = DataFrame(columns=['initial', 'pmin', 'pmax',
                                             'vary', 'name'])
        self.set_init_parameters()

        # initialize attributes
        self.masked_observations = None
        self.fit = None

        self.name = validate_name(name)

        # File Information
        self.file_info = self._get_file_info()

        # add plots
        self.plots = MetranPlot(self)

    @property
    def nparam(self):
        return self.parameters.index.size

    @property
    def nstate(self):
        return self.nseries + self.nfactors

    def standardize(self, oseries):
        """Method to standardize series.

        Standardized by subtracting mean and dividing by standard deviation.

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
        pairs = Series(index=oseries.columns, dtype=int)
        oseries["count"] = oseries.count(axis=1)
        for s in pairs.index:
            pairs[s] = oseries.dropna(subset=[s, ])["count"].count()
        oseries = oseries.drop(["count"], axis=1)
        if pairs.min() < max(min_pairs, 1):
            err = pairs[pairs < min_pairs].index.tolist()
            msg = "Number of cross-sectional data is less than " \
                + str(min_pairs) + " for series " \
                + (', ').join([str(e) for e in err])
            logger.error(msg)
            raise Exception(msg)

    def get_factors(self, oseries=None):
        """Method to get factor loadings based on factor analysis.

        This method also gets some relevant results from the factor analysis
        including the eigenvalues and percentage explained by factors (fep).

        Parameters
        ----------
        oseries : pandas.DataFrame, optional
            Series to be analyzed. The default is None.

        Returns
        -------
        factors : numpy.ndarray
            Factor loadings as estimated using factor analysis
        """
        if oseries is None:
            oseries = self.oseries
        fa = FactorAnalysis()
        self.factors = fa.solve(oseries)
        self.eigval = fa.eigval
        if self.factors is not None:
            self.nfactors = self.factors.shape[1]
            self.fep = fa.fep
        else:
            self.nfactors = 0

        return self.factors

    def _init_kalmanfilter(self, oseries):
        """Internal method, initialize Kalmanfilter for sequential processing.

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
        """Internal method to calculate autoregressive model parameter.

        Autoregressive model parameter is calculated based on parameter
        alpha.

        Parameters
        ----------
        alpha : float
            model parameter

        Returns
        -------
        float
            autoregressive model parameter
        """
        a = to_offset(self.settings["freq"]).delta / Timedelta(1, "D")
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
            name = self.snames[n] + "_sdf" + "_alpha"
            transition_matrix[n, n] = self._phi(p[name])
        for n in range(self.nfactors):
            name = "cdf" + str(n + 1) + "_alpha"
            transition_matrix[self.nseries + n,
                              self.nseries + n] = self._phi(p[name])
        return transition_matrix

    def get_transition_covariance(self, p=None, initial=False):
        """Get transition covariance matrix of the Metran dynamic factor model.

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
        factor_load = np.sum(np.square(self.factors), axis=1)
        for n in range(self.nseries):
            name = self.snames[n] + "_sdf" + "_alpha"
            transition_covariance[n, n] = (1 - self._phi(p[name]) ** 2) \
                * (1 - factor_load[n])
        for n in range(self.nfactors):
            name = "cdf" + str(n + 1) + "_alpha"
            transition_covariance[self.nseries + n, self.nseries + n] = (
                1 - self._phi(p[name]) ** 2
            )
        return transition_covariance

    def get_transition_variance(self, p=None, initial=False):
        """Get the transition variance vector.

        The transition variance vector is obtained by extracting the diagonal
        of the transition covariance matrix.

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
        """Internal method to get all matrices.

        Returns all matrices required to define the Metran dynamic
        factor model.

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
        None
        """
        pinit_alpha = 10
        for n in range(self.nfactors):
            self.parameters.loc["cdf" + str(n + 1) + "_alpha"] = (
                pinit_alpha, 1e-5, None, True, "cdf")
        for n in range(self.nseries):
            self.parameters.loc[self.snames[n] + "_sdf" + "_alpha"] = (
                pinit_alpha, 1e-5, None, True, "sdf")

    def mask_observations(self, mask):
        """Mask observations for processing with Kalman filter or smoother.

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
            self.masked_observations = self.oseries.mask(mask.astype(bool))
            self.kf.init_states()
            self.kf.set_observations(self.masked_observations)
            self.kf.mask = True

    def unmask_observations(self):
        """Method to unmask observation and reset observations.

        Returns
        -------
        None
        """
        self.masked_observations = None
        self.kf.init_states()
        self.kf.set_observations(self.oseries)
        self.kf.mask = False

    def set_observations(self, oseries):
        """Rework oseries to pandas.DataFrame for further use in Metran class.

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
                for i, os in enumerate(oseries):
                    if isinstance(os, TimeSeries):
                        _oseries.append(os.series)
                        _names.append(os.name)
                    elif isinstance(os, (Series, DataFrame)):
                        if isinstance(os, DataFrame):
                            if os.shape[1] > 1:
                                msg = "One or more series have " \
                                    + "DataFrame with multiple columns"
                                logger.error(msg)
                                raise Exception(msg)
                            os = os.squeeze()
                        if os.name is None:
                            os.name = 'Series' + str(i + 1)
                        _oseries.append(os)
                        _names.append(os.name)
                self.snames = _names
                oseries = concat(_oseries, axis=1)
            else:
                oseries = DataFrame()
        elif isinstance(oseries, DataFrame):
            self.snames = oseries.columns
        else:
            msg = "Input type should be either a " \
                + "list, tuple, or pandas.DataFrame"
            logger.error(msg)
            raise TypeError(msg)

        if oseries.shape[1] < 2:
            msg = "Metran requires at least 2 series, found " \
                + str(oseries.shape[1])
            logger.error(msg)
            raise Exception(msg)

        oseries = self.truncate(oseries)
        if isinstance(oseries.index, DatetimeIndex):
            oseries = oseries.asfreq("D")
            self.nseries = oseries.shape[1]
            self.oseries_unstd = oseries
            self.oseries = self.standardize(oseries)
            self.test_cross_section()
        else:
            msg = "Index of series must be DatetimeIndex"
            logger.error(msg)
            raise TypeError(msg)

    def get_observations(self, standardized=False, masked=False):
        """Returns series as available in Metran class.

        Parameters
        ----------
        standardized : bool, optional
            If True, obtain standardized observations. If False,
            obtain unstandardized observations. The default is False.
        masked : boolean
            If True, return masked observations. The default is False.

        Returns
        -------
        pandas.DataFrame
            Time series.
        """
        if masked:
            oseries = self.masked_observations
        else:
            oseries = self.oseries
        if not standardized:
            oseries = oseries * self.oseries_std + self.oseries_mean
        return oseries

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

    def get_specificity(self):
        """Get fraction that is explained by the specific dynamic factor.

        Calculate specificity for each series. The specificity is
        equal to (1 - communality).

        Returns
        -------
        numpy.ndarray
            For each series the specificity, a value between 0 and 1.
            A value of 0 means that the series has all variation
            in common with other series. A value of 1 means that the
            series has no variation in common.
        """
        return (1 - self.get_communality())

    def get_communality(self):
        """Get fraction that is explained by the common dynamic factor(s).

        Calculate communality for each series.

        Returns
        -------
        numpy.ndarray
            For each series the communality, a value between 0 and 1.
            A value of 0 means that the series has no variation
            in common with other series. A value of 1 means that the
            series has all variation in common.
        """
        return np.sum(np.square(self.factors), axis=1)

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
        columns = [name + "_sdf" for name in self.snames]
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
        columns = [name + "_sdf" for name in self.snames]
        columns.extend(["cdf" + str(i + 1) for i in range(self.nfactors)])
        state_variances = DataFrame(var, index=self.oseries.index,
                                    columns=columns)
        return state_variances

    def get_state(self, i, p=None, alpha=0.05, method="smoother"):
        """Get filtered or smoothed mean for specific state.

        Optionally including the 1-alpha confidence interval.

        Parameters
        ----------
        i : int
            index of state vector to be obtained
        p : pandas.Series
            Model parameters. The default is None.
        alpha : float, optional
            Include (1-alpha) confidence interval in DataFrame. The value
            of alpha must be between 0 and 1.
            If None, no confidence interval is returned. The default is 0.05.
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
            if alpha is not None:
                if alpha > 0 and alpha < 1:
                    z = norm.ppf(1 - alpha / 2.)
                else:
                    msg = "The value of alpha must be between 0 and 1."
                    logger.error(msg)
                    raise Exception(msg)
                variances = self.get_state_variances(
                    p=p, method=method).iloc[:, i]
                iv = z * np.sqrt(variances)
                state = concat([state, state - iv, state + iv], axis=1)
                state.columns = ['mean', 'lower', 'upper']
        return state

    def get_simulated_means(self, p=None, standardized=False,
                            method="smoother"):
        """Method to calculate simulated means.

        Simulated means are the filtered/smoothed mean estimates for
        the observed series.

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
        simulated_means : pandas.DataFrame
            Filtered or smoothed estimates for observed series.
        """
        self._run_kalman(method, p=p)
        if standardized:
            observation_matrix = self.get_observation_matrix(p=p)
            observation_means = np.zeros(observation_matrix.shape[0])
        else:
            observation_matrix = self.get_scaled_observation_matrix(p=p)
            observation_means = self.oseries_mean
        (means, _) = \
            self.kf.simulate(observation_matrix, method=method)
        simulated_means = \
            DataFrame(means, index=self.oseries.index,
                      columns=self.oseries.columns) + observation_means
        return simulated_means

    def get_simulated_variances(self, p=None, standardized=False,
                                method="smoother"):
        """Method to calculate simulated variances,

        The simulated variances are the filtered/smoothed variances
        for the observed series.

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
        simulated_variances : pandas.DataFrame
            Filtered or smoothed variances for observed series.
        """
        self._run_kalman(method, p=p)
        if standardized:
            observation_matrix = self.get_observation_matrix(p=p)
        else:
            observation_matrix = self.get_scaled_observation_matrix(p=p)
        (_, variances) = \
            self.kf.simulate(observation_matrix, method=method)
        simulated_variances = \
            DataFrame(variances, index=self.oseries.index,
                      columns=self.oseries.columns)
        return simulated_variances

    def get_simulation(self, name, p=None, alpha=0.05, standardized=False,
                       method="smoother"):
        """Method to calculate simulated means for specific series.

        Optionally including 1-alpha confidence interval.

        Parameters
        ----------
        name : str
            name of series to be obtained
        p : pandas.Series
            Model parameters. The default is None.
        alpha : float, optional
            Include (1-alpha) confidence interval in DataFrame. The value
            of alpha must be between 0 and 1.
            If None, no confidence interval is returned. The default is 0.05.
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
        sim = None
        means = self.get_simulated_means(p=p, standardized=standardized,
                                         method=method)
        if name in means.columns:
            sim = means.loc[:, name]
            if alpha is not None:
                if alpha > 0 and alpha < 1:
                    z = norm.ppf(1 - alpha / 2.)
                else:
                    msg = "The value of alpha must be between 0 and 1."
                    logger.error(msg)
                    raise Exception(msg)
                variances = \
                    self.get_simulated_variances(p=p,
                                                 standardized=standardized,
                                                 method=method).loc[:, name]
                iv = z * np.sqrt(variances)
                sim = concat([sim, sim - iv, sim + iv], axis=1)
                sim.columns = ['mean', 'lower', 'upper']
        else:
            logger.error("Unknown name: " + name)
        return sim

    def decompose_simulation(self, name, p=None, standardized=False,
                             method="smoother"):
        """Decompose simulation into specific and common dynamic components.

        Method to get for observed series filtered/smoothed estimate
        decomposed into specific dynamic component (sdf) and the sum of common
        dynamic components (cdf).

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
            observation_matrix = self.get_observation_matrix(p=p)
            observation_means = np.zeros(observation_matrix.shape[0])
        else:
            observation_matrix = self.get_scaled_observation_matrix(p=p)
            observation_means = self.oseries_mean
        (sdf_means, cdf_means) = \
            self.kf.decompose(observation_matrix, method=method)
        if name in self.oseries.columns:
            sdf = DataFrame(sdf_means,
                            index=self.oseries.index,
                            columns=self.oseries.columns) \
                + observation_means[self.oseries.columns.tolist().index(name)]
            df = sdf.loc[:, name]
            cols = ["sdf", ]
            for k in range(self.nfactors):
                cdf = DataFrame(cdf_means[k],
                                index=self.oseries.index,
                                columns=self.oseries.columns)
                df = concat([df, cdf.loc[:, name]], axis=1)
                cols.append("cdf" + str(k + 1))
            df.columns = cols
        else:
            logger.error("Unknown name: " + name)
        return df

    def get_scaled_observation_matrix(self, p=None):
        """Method scale observation matrix by standard deviations of oseries.

        Returns
        -------
        observation_matrix: numpy.ndarray
            scaled observation matrix
        p : pandas.Series
            Model parameters. The default is None.
        """
        scale = self.oseries_std
        observation_matrix = self.get_observation_matrix(p=p)
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
        factors = self.get_factors(self.oseries)
        if factors is not None:
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
        except Exception:
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
        w = max(width - 45, 0)
        header = "Fit report {name:<16}{string}Fit Statistics\n" \
                 "{line}\n".format(name=self.name[:14],
                                   string=string.format(
                                       "", fill=' ', align='>', width=w),
                                   line=string.format("", fill='=', align='>',
                                                      width=width))

        basic = ""
        vw = max(width - 45, 0)
        for (val1, val2), (val3, val4) in zip(model.items(), fit.items()):
            val4 = string.format(val4, fill=' ', align='>', width=w)
            space = string.format("", fill=' ', align='>', width=vw)
            basic += "{:<8} {:<16} {:} {:<7} {:}\n".format(val1, val2, space,
                                                           val3, val4)

        # Create the parameters block
        parameters = "\nParameters ({n_param} were optimized)\n{line}\n" \
                     "{parameters}".format(n_param=parameters.vary.sum(),
                                           line=string.format("", fill='=',
                                                              align='>',
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

        # Create the state parameters block
        phi = np.diag(self.get_transition_matrix())
        q = self.get_transition_variance()
        names = [name + "_sdf" for name in self.snames]
        names.extend(["cdf" + str(i + 1) for i in range(self.nfactors)])
        transition = DataFrame(np.array([phi, q]).T,
                               index=names,
                               columns=["phi", "q"])

        # get width of index to align state parameters index
        idx_width = int(max([len(n) for n in transition.index]))

        # Create the communality block
        communality = Series(self.get_communality(),
                             index=self.oseries.columns,
                             name="")
        communality.index = [idx.ljust(idx_width)
                             for idx in communality.index]
        communality = communality.apply("{:.2%}".format).to_frame()

        # Create the observation parameters block
        gamma = self.factors
        names = ["gamma" + str(i + 1) for i in range(self.nfactors)]
        observation = DataFrame(gamma,
                                index=self.oseries.columns,
                                columns=names)
        observation.index = [idx.ljust(idx_width)
                             for idx in observation.index]
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
