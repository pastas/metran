"""This module contains the Kalman filter class for Metran and associated
filtering and smoothing methods."""

import sys
from logging import getLogger

import numpy as np
from pastas.decorators import njit
from pastas.utils import initialize_logger

logger = getLogger(__name__)
initialize_logger(logger)


def filter_predict(filtered_state_mean, filtered_state_covariance,
                   transition_matrix, transition_covariance):
    """Predict state with a Kalman Filter using sequential processing.

    Parameters
    ----------
    filtered_state_mean: numpy.ndarray
        Mean of state at time t-1 given observations from times
        [0...t-1]
    filtered_state_covariance: numpy.ndarray
        Covariance of state at time t-1 given observations from times
        [0...t-1]

    Returns
    -------
    predicted_state_mean : numpy.ndarray
        Mean of state at time t given observations from times [0...t-1]
    predicted_state_covariance : numpy.ndarray
        Covariance of state at time t given observations from times
        [0...t-1]
    """
    predicted_state_mean = np.dot(transition_matrix, filtered_state_mean)
    predicted_state_covariance = (np.dot(transition_matrix,
                                         np.dot(filtered_state_covariance,
                                                transition_matrix.T))
                                  + transition_covariance)

    return (predicted_state_mean, predicted_state_covariance)


def filter_update(observations, observation_matrix, observation_variance,
                  observation_indices, observation_count,
                  state_mean, state_covariance):
    """Update predicted state with Kalman Filter using sequential processing.

    Parameters
    ----------
    observations : numpy.ndarray
        Observations for sequential processing of Kalman filter.
    observation_matrix : numpy.ndarray
        observation matrix to project state.
    observation_variance : numpy.ndarray
        observation variances
    observation_indices : numpy.ndarray
        used to compress observations, observation_matrix,
        and observation_variance skipping missing values.
    observation_count : numpy.ndarray
        number of observed time series for each timestep
        determining the number of elements to be read in observation_indices.
    state_mean : numpy.ndarray
        mean of state at time t given observations from times
        [0...t-1]
    state_covariance : numpy.ndarray
        covariance of state at time t given observations from times
        [0...t-1]

    Returns
    -------
    state_mean : [n_dim_state] array
        Mean of state at time t given observations from times
        [0...t], i.e. updated state mean
    state_covariance : [n_dim_state, n_dim_state] array
        Covariance of state at time t given observations from times
        [0...t], i.e. updated state covariance
    sigma : float
        Weighted squared innovations.
    detf : float
        Log of determinant of innovation variances matrix.
    """

    sigma = 0.
    detf = 0.
    n_observation = np.int(observation_count)
    for i in range(n_observation):
        observation_index = int(observation_indices[i])
        obsmat = observation_matrix[observation_index, :]
        innovation = (observations[observation_index]
                      - np.dot(obsmat, state_mean))
        dot_statecov_obsmat = np.dot(state_covariance, obsmat)
        innovation_covariance = (np.dot(obsmat, dot_statecov_obsmat)
                                 + observation_variance[observation_index])
        kgain = dot_statecov_obsmat / innovation_covariance
        state_covariance = (state_covariance
                            - (np.outer(kgain, kgain)
                               * innovation_covariance))

        state_mean = state_mean + kgain * innovation

        sigma = sigma + (innovation ** 2 / innovation_covariance)
        detf = detf + np.log(innovation_covariance)

    return (state_mean, state_covariance, sigma, detf)


def seqkalmanfilter_np(observations, transition_matrix, transition_covariance,
                       observation_matrix, observation_variance,
                       observation_indices, observation_count,
                       filtered_state_mean, filtered_state_covariance):
    """Method to run sequential Kalman filter optimized for use with numpy.

    This method is suggested if numba is not installed.
    It is, however, much slower than seqkalmanfilter combined with numba.

    Parameters
    ----------
    observations : numpy.ndarray
        Observations for sequential processing of Kalman filter.
    transition_matrix : numpy.ndarray
        State transition matrix from time t-1 to t.
    transition_covariance : numpy.ndarray
        State transition covariance matrix from time t-1 to t.
    observation_matrix : numpy.ndarray
        Observation matrix to project state.
    observation_variance : numpy.ndarray
        Observation variances
    observation_indices : numpy.ndarray
        Used to compress observations, observation_matrix,
        and observation_variance skipping missing values.
    observation_count : numpy.ndarray
        Number of observed time series for each timestep
        determining the number of elements to be read in observation_indices.
    filtered_state_mean : numpy.ndarray
        Initial state mean
    filtered_state_covariance : numpy.ndarray
        Initial state covariance

    Returns
    -------
    sigmas : list
        Weighted squared innovations.
    detfs : list
        Log values of determinant of innovation variances matrix.
    filtered_state_means : list
        `filtered_state_means[t]` = mean state estimate
        for time t given observations from times [0...t].
    filtered_state_covariances : list
        `filtered_state_covariances[t]` = covariance of state estimate
        for time t given observations from times [0...t].
    predicted_state_means : list
        `predicted_state_means[t]` = mean state estimate
        for time t given observations from times [0...t-1].
    predicted_state_covariances : list
        `predicted_state_covariances[t]` = covariance of state estimate
        for time t given observations from times [0...t-1].
    """
    # initialization
    n_timesteps = int(observations.shape[0])
    dim = filtered_state_mean.shape[0]
    sigmas = []
    detfs = []
    filtered_state_means = np.zeros((n_timesteps, dim),
                                    dtype=np.float64)
    filtered_state_covariances = np.zeros((n_timesteps, dim, dim),
                                          dtype=np.float64)
    predicted_state_means = np.zeros((n_timesteps, dim),
                                     dtype=np.float64)
    predicted_state_covariances = np.zeros((n_timesteps, dim, dim),
                                           dtype=np.float64)
    for t in range(n_timesteps):
        # Kalman filter prediction step
        (predicted_state_mean, predicted_state_covariance) = (
            filter_predict(filtered_state_mean, filtered_state_covariance,
                           transition_matrix, transition_covariance))
        predicted_state_means[t] = predicted_state_mean
        predicted_state_covariances[t] = predicted_state_covariance

        if observation_count[t] > 0:
            # Kalman filter update step
            (filtered_state_mean,
             filtered_state_covariance,
             sigma, detf) = filter_update(observations[t, :],
                                          observation_matrix,
                                          observation_variance,
                                          observation_indices[t, :],
                                          observation_count[t],
                                          predicted_state_mean,
                                          predicted_state_covariance)
            sigmas.append(sigma)
            detfs.append(detf)
        else:
            filtered_state_mean = predicted_state_mean
            filtered_state_covariance = predicted_state_covariance

        filtered_state_means[t] = filtered_state_mean
        filtered_state_covariances[t] = filtered_state_covariance

    return (sigmas, detfs, len(sigmas),
            filtered_state_means,
            filtered_state_covariances,
            predicted_state_means,
            predicted_state_covariances)


@njit('( float64[:,:], float64[:,:], float64[:,:], \
      float64[:,:], float64[:], \
      float64[:,:], int64[:],   \
      float64[:], float64[:,:]  \
      )')
def seqkalmanfilter(observations, transition_matrix, transition_covariance,
                    observation_matrix, observation_variance,
                    observation_indices, observation_count,
                    filtered_state_mean, filtered_state_covariance):
    """Method to run sequential Kalman filter optimized for use with numba.

    This method requires numba to be installed. With numba, this method
    is much faster than seqkalmanfilter_np. However, without numba,
    it is extremely slow and seqkalmanfilter_np should be used.

    Parameters
    ----------
    observations : numpy.ndarray
        Observations for sequential processing of Kalman filter.
    transition_matrix : numpy.ndarray
        State transition matrix from time t-1 to t.
    transition_covariance : numpy.ndarray
        State transition covariance matrix from time t-1 to t.
    observation_matrix : numpy.ndarray
        Observation matrix to project state.
    observation_variance : numpy.ndarray
        Observation variances
    observation_indices : numpy.ndarray
        Used to compress observations, observation_matrix,
        and observation_variance skipping missing values.
    observation_count : numpy.ndarray
        Number of observed time series for each timestep
        determining the number of elements to be read in observation_indices.
    filtered_state_mean : numpy.ndarray
        Initial state mean
    filtered_state_covariance : numpy.ndarray
        Initial state covariance

    Returns
    -------
    sigmas : numpy.ndarray
        Weighted squared innovations.
    detfs : numpy.ndarray
        Log of determinant of innovation variances matrix.
    sigmacount : int
        Number of elements in sigmas en detfs with calculated values.
    filtered_state_means : numpy.ndarray
        `filtered_state_means[t]` = mean state estimate
        for time t given observations from times [0...t].
    filtered_state_covariances : numpy.ndarray
        `filtered_state_covariances[t]` = covariance of state estimate
        for time t given observations from times [0...t].
    predicted_state_means : numpy.ndarray
        `predicted_state_means[t]` = mean state estimate
        for time t given observations from times [0...t-1].
    predicted_state_covariances : numpy.ndarray
        `predicted_state_covariances[t]` = covariance of state estimate
        for time t given observations from times [0...t-1].
    """
    # initialization
    n_timesteps = observation_count.shape[0]
    dim = filtered_state_mean.shape[0]
    sigmas = np.zeros(n_timesteps, dtype=np.float64)
    detfs = np.zeros(n_timesteps, dtype=np.float64)
    filtered_state_means = np.zeros((n_timesteps, dim),
                                    dtype=np.float64)
    filtered_state_covariances = np.zeros((n_timesteps, dim, dim),
                                          dtype=np.float64)
    predicted_state_means = np.zeros((n_timesteps, dim),
                                     dtype=np.float64)
    predicted_state_covariances = np.zeros((n_timesteps, dim, dim),
                                           dtype=np.float64)
    sigmacount = 0

    for t in range(n_timesteps):
        predicted_state_mean = np.zeros(dim, dtype=np.float64)
        predicted_state_covariance = np.zeros((dim, dim), dtype=np.float64)
        for r in range(dim):
            summed = 0.
            for c in range(dim):
                summed += transition_matrix[r, c] * filtered_state_mean[c]
            predicted_state_mean[r] = summed

        for r in range(dim):
            for c in range(dim):
                predicted_state_covariance[r, c] = (
                    transition_matrix[r, r] * filtered_state_covariance[r, c]
                    * transition_matrix[c, c] + transition_covariance[r, c])
        predicted_state_means[t] = predicted_state_mean
        predicted_state_covariances[t] = predicted_state_covariance

        if observation_count[t] > 0:
            sigma = 0.
            detf = 0.
            filtered_state_mean = np.zeros(dim, dtype=np.float64)
            filtered_state_covariance = np.zeros((dim, dim), dtype=np.float64)

            for i in range(observation_count[t]):
                idx = np.int64(observation_indices[t, i])

                summed = 0.
                for r in range(dim):
                    summed += (observation_matrix[idx, r]
                               * predicted_state_mean[r])
                innovation = observations[t, idx] - summed

                dotmat = np.zeros(dim, dtype=np.float64)
                for r in range(dim):
                    summed = 0.
                    for c in range(dim):
                        summed += (predicted_state_covariance[r, c]
                                   * observation_matrix[idx, c])
                    dotmat[r] = summed

                summed = 0.
                for r in range(dim):
                    summed += observation_matrix[idx, r] * dotmat[r]
                innovation_variance = observation_variance[idx] + summed

                kgain = np.zeros(dim, dtype=np.float64)
                for r in range(dim):
                    kgain[r] = dotmat[r] / innovation_variance

                for r in range(dim):
                    for c in range(dim):
                        predicted_state_covariance[r, c] +=           \
                            - kgain[r] * kgain[c] * innovation_variance

                for r in range(dim):
                    predicted_state_mean[r] += kgain[r] * innovation

                sigma += (innovation ** 2 / innovation_variance)
                detf += np.log(innovation_variance)

            sigmas[sigmacount] = sigma
            detfs[sigmacount] = detf
            sigmacount += 1

        for r in range(dim):
            filtered_state_mean[r] = predicted_state_mean[r]
        for r in range(dim):
            for c in range(dim):
                filtered_state_covariance[r, c] = \
                    predicted_state_covariance[r, c]
        filtered_state_means[t] = filtered_state_mean
        filtered_state_covariances[t] = filtered_state_covariance

    return (sigmas, detfs, sigmacount,
            filtered_state_means,
            filtered_state_covariances,
            predicted_state_means,
            predicted_state_covariances)


def kalmansmoother(filtered_state_means, filtered_state_covariances,
                   predicted_state_means, predicted_state_covariances,
                   transition_matrix):
    """Method to run the Kalman smoother.

    Estimate the hidden state at time for each time step given all
    observations.

    Parameters
    ----------
    filtered_state_means : array_like
        `filtered_state_means[t]` = mean state estimate
        for time t given observations from times [0...t].
    filtered_state_covariances : array_like
        `filtered_state_covariances[t]` = covariance of state estimate
        for time t given observations from times [0...t].
    predicted_state_means : array_like
        `predicted_state_means[t]` = mean state estimate
        for time t given observations from times [0...t-1].
    predicted_state_covariances : array_like
        `predicted_state_covariances[t]` = covariance of state estimate
        for time t given observations from times [0...t-1].
    transition_matrix : numpy.ndarray
        State transition matrix from time t-1 to t.

    Returns
    -------
    smoothed_state_means : numpy.ndarray
        Mean of hidden state distributions
        for times [0...n_timesteps-1] given all observations
    smoothed_state_covariances : numpy.ndarray
        Covariance matrix of hidden state distributions
        for times [0...n_timesteps-1] given all observations
    """

    n_timesteps = len(filtered_state_means)
    n_state = len(filtered_state_means[0])

    smoothed_state_means = np.zeros((n_timesteps, n_state))
    smoothed_state_covariances = np.zeros((n_timesteps, n_state,
                                           n_state))

    kalman_smoothing_gains = np.zeros((n_timesteps - 1, n_state,
                                       n_state))

    smoothed_state_means[-1] = filtered_state_means[-1]
    smoothed_state_covariances[-1] = filtered_state_covariances[-1]

    for t in reversed(range(n_timesteps - 1)):
        try:
            psc_inv = np.linalg.pinv(predicted_state_covariances[t + 1])
        except Exception:
            psc_inv = np.linalg.inv(predicted_state_covariances[t + 1])
        kalman_smoothing_gains[t] = (np.dot(filtered_state_covariances[t],
                                            np.dot(transition_matrix.T, psc_inv)))
        smoothed_state_means[t] = (filtered_state_means[t]
                                   + np.dot(kalman_smoothing_gains[t],
                                            (smoothed_state_means[t + 1]
                                             - predicted_state_means[t + 1])))
        smoothed_state_covariances[t] = (
            filtered_state_covariances[t] + np.dot(
                kalman_smoothing_gains[t],
                np.dot((smoothed_state_covariances[t + 1]
                        - predicted_state_covariances[t + 1]),
                       kalman_smoothing_gains[t].T)))

    return (smoothed_state_means, smoothed_state_covariances)


class SPKalmanFilter():
    """Kalman filter class for Metran.

    Parameters
    ----------
    engine : str, optional
        Engine to be used to run sequential Kalman filter.
        Either "numba" or "numpy". The default is "numba".

    Returns
    -------
    kf: kalmanfilter.SPKalmanFilter
        Metran SPKalmanfilter instance.
    """

    def __init__(self, engine="numba"):
        self.init_states()
        self.detfs = None
        self.sigmas = None
        self.nobs = None
        self.mask = False

        if engine == "numpy" or "numba" not in sys.modules:
            self.filtermethod = seqkalmanfilter_np
        else:
            self.filtermethod = seqkalmanfilter

    def init_states(self):
        """Method to initialize state means and covariances.

        Returns
        -------
        None.
        """
        self.filtered_state_means = None
        self.filtered_state_covariances = None
        self.predicted_state_means = None
        self.predicted_state_covariances = None
        self.smoothed_state_means = None
        self.smoothed_state_covariances = None

    def set_matrices(self, transition_matrix, transition_covariance,
                     observation_matrix, observation_variance):
        """Method to set matrices of state space model.

        Parameters
        ----------
        transition_matrix : numpy.ndarray
            State transition matrix
        transition_covariance : numpy.ndarray
            State transition covariance matrix.
        observation_matrix : numpy.ndarray
            Observation matrix.
        observation_variance : numpy.ndarray
            Observation variance.

        Returns
        -------
        None.
        """
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        self.observation_matrix = observation_matrix
        self.observation_variance = observation_variance
        self.nstate = np.int64(self.transition_matrix.shape[0])

    def get_mle(self, warmup=1):
        """Method to calculate maximum likelihood estimate.

        Parameters
        ----------
        warmup : int, optional
            Number of time steps to skip. The default is 1.

        Returns
        -------
        mle : float
            Maximum likelihood estimate.
        """
        detfs = self.detfs[warmup:]
        sigmas = self.sigmas[warmup:]
        nobs = np.sum(self.observation_count[warmup:])
        mle = nobs * np.log(2 * np.pi) + np.sum(detfs) + np.sum(sigmas)
        return mle

    def simulate(self, observation_matrix, method="smoother"):
        """Method to get simulated means and covariances.

        Parameters
        ----------
        observation_matrix : numpy.ndarray
            Observation matrix for projecting states.
        method : str, optional
            If "filter", use Kalman filter to obtain estimates.
            If "smoother", use Kalman smoother. The default is "smoother".

        Returns
        -------
        simulated_means : list
            List of simulated means for each time step.
        simulated_variances : list
            List of simulated variances for each time step.
            Variances are diagonal elements of simulated covariance matrix.
        """
        if method == "filter":
            means = self.filtered_state_means
            covariances = self.filtered_state_covariances
        else:
            means = self.smoothed_state_means
            covariances = self.smoothed_state_covariances
        simulated_means = []
        simulated_variances = []
        for t, _ in enumerate(means):
            simulated_means.append(np.dot(observation_matrix, means[t]))
            var = np.diag(np.dot(observation_matrix,
                                 np.dot(covariances[t], observation_matrix.T)))
            # prevent variances to become less than 0
            simulated_variances.append(np.maximum(var, 0))
        return (simulated_means, simulated_variances)

    def decompose(self, observation_matrix, method="smoother"):
        """Method to decompose simulated means.

        Decomposition into specific dynamic factors (sdf) and common
        dynamic factors (cdf).

        Parameters
        ----------
        observation_matrix : numpy.ndarray
            Observation matrix for projecting states.
        method : str, optional
            If "filter", use Kalman filter to obtain estimates.
            If "smoother", use Kalman smoother. The default is "smoother".

        Returns
        -------
        sdf_means : list
            List of specific dynamic factors for each time step.
        cdf_means : list
            List of common dynamic factor(s) for each time step.
        """
        if method == "filter":
            means = self.filtered_state_means
        else:
            means = self.smoothed_state_means
        nsdf = self.observation_matrix.shape[0]
        ncdf = self.observation_matrix.shape[1] - nsdf
        sdf_means = []
        for t, _ in enumerate(means):
            sdf_means.append(np.dot(observation_matrix[:, :nsdf],
                                    means[t, :nsdf]))
        cdf_means = [[]] * ncdf
        for k in range(ncdf):
            idx = nsdf + k
            for t, _ in enumerate(means):
                cdf_means[k].append(np.dot(observation_matrix[:, idx],
                                           means[t, idx]))
        return (sdf_means, cdf_means)

    def set_observations(self, oseries):
        """Construct observation matrices allowing missing values.

        Initialize sequential processing of the Kalman filter.

        Parameters
        ----------
        oseries : pandas.DataFrame
            multiple time series
        """
        self.oseries_index = oseries.index
        observations_masked = np.ma.array(oseries,
                                          mask=(~np.isfinite(oseries)))
        (n_timesteps, dimobs) = observations_masked.shape
        self.observation_indices = np.zeros((n_timesteps, dimobs),
                                            dtype=np.float64)
        self.observation_count = np.zeros(n_timesteps, dtype=np.int64)
        self.observations = np.zeros((n_timesteps, dimobs), dtype=np.float64)

        for t in range(n_timesteps):
            observation = observations_masked[t]
            # add large value to find all finite non-masked values
            obstmp = observation + 1e10
            obsindices = obstmp.nonzero()[0]
            self.observation_count[t] = len(obsindices)

            if (len(obsindices) > 0):
                for i, _ in enumerate(obsindices):
                    obsid = int(obsindices[i])
                    self.observations[t, obsid] = observation[obsid]
                    self.observation_indices[t, i] = obsid

    def run_smoother(self):
        """Run Kalman smoother.

        Calculate smoothed state means and covariances using the Kalman
        smoother.
        """
        # run Kalman filter to get filtered state estimates and covariances
        self.run_filter()
        # run Kalman smoother to get smoothed state estimates and covariances
        (smoothed_state_means,
         smoothed_state_covariances) = \
            kalmansmoother(self.filtered_state_means,
                           self.filtered_state_covariances,
                           self.predicted_state_means,
                           self.predicted_state_covariances,
                           self.transition_matrix)

        self.smoothed_state_means = smoothed_state_means
        self.smoothed_state_covariances = smoothed_state_covariances

    def run_filter(self, initial_state_mean=None,
                   initial_state_covariance=None,
                   engine=None):
        """Method to run the Kalman Filter.

        This is a sequential processing implementation of the Kalman filter
        requiring a diagonal observation error covariance matrix.
        The algorithm allows for missing data using the arrays
        observation_count giving the number of observations for each timestep,
        and observation_indices containing the corresponding indices
        of those observations used to select the appropriate rows
        from observation_matrix and observation_variance.
        These arrays have been constructed with self.set_observations()

        Parameters
        ----------
        initial_state_mean : array_like
            state vector for initializing Kalman filter.
        initial_state_covariance : array_like
            state covariance matrix for initializing Kalman filter.
        engine : str, optional
            Engine to be used to run sequential Kalman filter.
            Either "numba" or "numpy". The default is None, which
            means that the default Class setting is used.

        Returns
        -------
        None
        """

        if self.mask:
            logger.info("Running Kalman filter with masked observations.")

        if engine is not None:
            if engine == "numpy":
                self.filtermethod = seqkalmanfilter_np
            elif engine == "numba":
                if "numba" not in sys.modules:
                    msg = "Numba is not installed. Please install numba " \
                        "or use engine=numpy."
                    logger.error(msg)
                    raise Exception(msg)
                else:
                    self.filtermethod = seqkalmanfilter
            else:
                msg = "Unknown engine defined in run_filter."
                logger.error(msg)
                raise Exception(msg)

        if initial_state_mean is None:
            initial_state_mean = np.zeros(self.nstate)
        if initial_state_covariance is None:
            initial_state_covariance = np.eye(self.nstate)

        # Kalman filter
        (sigmas, detfs, sigmacount,
         filtered_state_means,
         filtered_state_covariances,
         predicted_state_means,
         predicted_state_covariances) = \
            self.filtermethod(self.observations,
                              self.transition_matrix,
                              self.transition_covariance,
                              self.observation_matrix,
                              self.observation_variance,
                              self.observation_indices,
                              self.observation_count,
                              initial_state_mean,
                              initial_state_covariance)

        self.sigmas = sigmas[:sigmacount]
        self.detfs = detfs[:sigmacount]
        self.filtered_state_means = filtered_state_means
        self.filtered_state_covariances = filtered_state_covariances
        self.predicted_state_means = predicted_state_means
        self.predicted_state_covariances = predicted_state_covariances
