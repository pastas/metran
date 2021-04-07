# -*- coding: utf-8 -*-
"""Created on Mon Mar 29 08:47:25 2021.

@author: Wilbert Berendrecht
"""

import importlib
import numpy as np
from pandas import DataFrame
from pastas.decorators import njit
from logging import getLogger

logger = getLogger(__name__)

def filter_predict(current_state_mean, current_state_covariance,
                   transition_matrix, transition_covariance):
    """Calculate the mean and covariance of :math:`P(x_{t|t-1})`

    Parameters
    ----------
    current_state_mean: [n_dim_state] array
        mean of state at time t given observations from times
        [0...t-1]
    current_state_covariance: [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from times
        [0...t-1]

    Returns
    -------
    predicted_state_mean : [n_dim_state] array
        mean of state at time t given observations from times [0...t-1]
    predicted_state_covariance : [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from times
        [0...t-1]
    """
    predicted_state_mean = np.dot(transition_matrix, current_state_mean)
    predicted_state_covariance = (np.dot(transition_matrix,
                                         np.dot(current_state_covariance,
                                                transition_matrix.T)
                                         )
                                  + transition_covariance
    )

    return (predicted_state_mean, predicted_state_covariance)

def filter_update(observations, observation_matrix, observation_variance,
                  observation_indices, observation_count,
                  predicted_state_mean, predicted_state_covariance):
    """Update a predicted state with a Kalman Filter update using
    sequential processing (assumption of uncorrelated observation error)

    Parameters
    ----------
    predicted_state_mean : [n_dim_state] array
        mean of state at time t given observations from times
        [0...t-1], i.e. predicted state mean
    predicted_state_covariance : [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from times
        [0...t-1], i.e. predicted state covariance

    Returns
    -------
    filtered_state_mean : [n_dim_state] array
        mean of state at time t given observations from times
        [0...t], i.e. updated state mean
    filtered_state_covariance : [n_dim_state, n_dim_state] array
        covariance of state at time t given observations from times
        [0...t], i.e. updated state covariance
    sigma : float
        :math:`\\nu_{t}^{2}/f_{t}` to be used in
        concentrated loglikelihood
    detf : float
        determinant of :math:`F_{t}` to be used in
        concentrated loglikelihood
    """

    sigma = 0.
    detf = 0.
    n_observation = np.int(observation_count)
    for i in range(n_observation):
        observation_index = int(observation_indices[i])
        obsmat = observation_matrix[observation_index, :]
        innovation = (observations[observation_index]
                      - np.dot(obsmat, predicted_state_mean))
        dot_statecov_obsmat = np.dot(predicted_state_covariance, obsmat)
        innovation_covariance = (np.dot(obsmat, dot_statecov_obsmat)
                                 + observation_variance[observation_index])
        kgain = dot_statecov_obsmat / innovation_covariance
        predicted_state_covariance = (predicted_state_covariance
                                      - (np.outer(kgain, kgain)
                                         * innovation_covariance))

        predicted_state_mean = predicted_state_mean + kgain * innovation

        sigma = sigma + (innovation**2 / innovation_covariance)
        detf = detf + np.log(innovation_covariance)


    return (predicted_state_mean, predicted_state_covariance, sigma, detf)


def seqkalmanfilter_np(observations, transition_matrix, transition_covariance,
                       observation_matrix, observation_variance,
                       observation_indices, observation_count,
                       filtered_state_mean, filtered_state_covariance):
    """
    Apply the Kalman Filter at time :math:`t = [0,...,N]`
    given observations up to and including time `t`.

    """
    # get number of time steps as number of Kalman filter recursions
    n_timesteps = int(observations.shape[0])

    # initialization
    sigmas = []
    detfs = []

    # initialize Kalman filter states and covariances
    filtered_state_means = []
    filtered_state_covariances = []

    for t in range(n_timesteps):
        # Kalman filter prediction step
        (predicted_state_mean, predicted_state_covariance) = (
            filter_predict(filtered_state_mean, filtered_state_covariance,
                           transition_matrix, transition_covariance)
        )

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
            # construct list of values used for likelihood
            sigmas.append(sigma)
            detfs.append(detf)
        else:
            filtered_state_mean = predicted_state_mean
            filtered_state_covariance = predicted_state_covariance

        filtered_state_means.append(filtered_state_mean)
        filtered_state_covariances.append(filtered_state_covariance)

    return (sigmas, detfs, len(sigmas), filtered_state_means,
            filtered_state_covariances)

@njit('( float64[:,:], float64[:,:], float64[:,:], \
      float64[:,:], float64[:], \
      float64[:,:], int64[:],   \
      float64[:], float64[:,:]  \
      )')
def seqkalmanfilter(observations, transition_matrix, transition_covariance,
                    observation_matrix, observation_variance,
                    observation_indices, observation_count,
                    filtered_state_mean, filtered_state_covariance):

    # initialization
    n_timesteps = observation_count.shape[0]
    dim = filtered_state_mean.shape[0]
    sigmas = np.zeros(n_timesteps, dtype=np.float64)
    detfs = np.zeros(n_timesteps, dtype=np.float64)
    filtered_state_means = np.zeros((n_timesteps, dim),
                                    dtype=np.float64)
    filtered_state_covariances = np.zeros((n_timesteps, dim, dim),
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
                   transition_matrix[r, r]
                   * filtered_state_covariance[r, c]
                   * transition_matrix[c, c]
                   + transition_covariance[r, c]
                   )

        if observation_count[t] > 0:
            sigma = 0.
            detf = 0.
            filtered_state_mean = np.zeros(dim, dtype=np.float64)
            filtered_state_covariance = np.zeros((dim, dim), dtype=np.float64)

            for i in range(observation_count[t]):
                idx = np.int64(observation_indices[t, i])

                summed = 0.
                for r in range(dim):
                    summed += observation_matrix[idx, r] \
                        * predicted_state_mean[r]
                innovation = observations[t, idx] - summed

                dotmat = np.zeros(dim, dtype=np.float64)
                for r in range(dim):
                    summed = 0.
                    for c in range(dim):
                        summed += predicted_state_covariance[r, c] \
                            * observation_matrix[idx, c]
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

                sigma += (innovation**2 / innovation_variance)
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

    return (sigmas, detfs, sigmacount, filtered_state_means,
            filtered_state_covariances)


class SPKalmanFilter():

    def __init__(self, engine="numba"):
        """
        """

        self.filtered_state_means = None
        self.filtered_state_covariances = None
        self.predicted_state_means = None
        self.predicted_state_covariances = None
        self.detfs = None
        self.sigmas = None
        self.nobs = None

        if engine == "fortran":
            from KalmanFilterf95 import kfseq
            self.filtermethod = kfseq
        elif engine == "numba":
            check_numba = importlib.util.find_spec("numba")
            if check_numba is not None and check_numba.name == "numba":
                self.filtermethod = seqkalmanfilter
            else:
                 logger.warning("Numba is not installed. Installing Numba is "
                                "recommended for significant speed-ups.")
                 self.filtermethod = seqkalmanfilter_np
        else:
            self.filtermethod = seqkalmanfilter_np


    def set_matrices(self, transition_matrix, transition_covariance,
                     observation_matrix, observation_variance):
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        self.observation_matrix = observation_matrix
        self.observation_variance = observation_variance
        self.nstate = np.int64(self.transition_matrix.shape[0])

    def get_mle(self, warmup=1):
        detfs = self.detfs[warmup:]
        sigmas = self.sigmas[warmup:]
        nobs = np.sum(self.observation_count[warmup:])
        mle = nobs * np.log(2*np.pi) + np.sum(detfs) + np.sum(sigmas)
        return mle

    def get_scale(self):
        return np.sum(self.sigmas) / self.nobs

    def get_smoothed_states_ci(self):
        if self.smoothed_states_mean is None:
            self.calculate_smoothed()
        ci_upper = self.smoothed_states_mean + 1.96 * self.smoothed_states_std
        ci_lower = self.smoothed_states_mean - 1.96 * self.smoothed_states_std
        ci = DataFrame(data=[ci_lower, ci_upper],
                       index=self.smoothed_states_mean.index,
                       columns=['lower', 'upper'])
        return ci

    def calculate_smoothed(self):

        """
        Get smoothed state estimate and confidence interval using the Kalman Smoother

        Parameters
        ----------
        residuals : pandas DataFrame
            residuals from univariate modelruns
        transition_matrix : array [n_dim_state, n_dim_state]
            transition matrix defined in :func:`setpar`
        transition_covariance : array [n_dim_state, n_dim_state]
            transition covariance matrix defined in :func:`setpar`
        observation_matrix : array [n_dim_obs, n_dim_state]
            observation matrix defined in :func:`construct_observation_matrices`
        observation_variance : array [n_dim_obs, n_dim_obs]
            observation covariance matrix defined in :func:`construct_observation_matrices`
        parmulti : dictionary
            initial parameter definition

        Returns
        -------
        smoothed : dictionary
            smoothed factors, smoothed factors projected on observed time series, standard deviations
        """


        # run Kalman filter to get filtered state estimates and covariances
        self.run_filter()
        # run Kalman smoother to get smoothed state estimates and covariances
        smoothed_state_means, smoothed_state_covariances = self._run_smoother()
        dfindex=self.oseries_index

        # means = np.vstack(smoothed_state_means[i] for i in range(self.nstate))
        with np.errstate(invalid='ignore'):
            std = np.vstack(np.sqrt(np.diag(smoothed_state_covariances[i]))
                            for i in range(self.nstate))
        self.smoothed_states_mean = DataFrame(smoothed_state_means,
                                              index=self.oseries_index)
        self.smoothed_states_std = DataFrame(std, index=self.oseries_index)

        _oseries_smoothed_means = []
        _oseries_smoothed_std = []
        for t in range(len(smoothed_state_means)):
            _oseries_smoothed_means.append(np.dot(self.observation_matrix,
                                                  smoothed_state_means[t]))
            var = np.diag(np.dot(self.observation_matrix,
                                 np.dot(smoothed_state_covariances[t],
                                        np.transpose(self.observation_matrix)
                                        )
                                 )
                          )
            _oseries_smoothed_std.append(np.sqrt(np.maximum(var, 0)))

        self.oseries_smoothed_means = DataFrame(_oseries_smoothed_means,
                                                index=dfindex)
        self.oseries_smoothed_std = DataFrame(_oseries_smoothed_std,
                                              index=dfindex)

    def _run_smoother(self):
        """
        Apply the Kalman Smoother

        Estimate the hidden state at time for each time step given all
        observations.

        Parameters
        ----------
        updated_state_means : list [n_timesteps]
            `updated_state_means[t]` = mean state estimate for time t given
            observations from times [0...t]
        updated_state_covariances : list [n_timesteps]
            `updated_state_covariances[t]` = covariance of state estimate for time
            t given observations from times [0...t]
        predicted_state_means : list [n_timesteps]
            `predicted_state_means[t]` = mean state estimate for time t given
            observations from times [0...t-1]
        predicted_state_covariances : list [n_timesteps]
            `predicted_state_covariances[t]` = covariance of state estimate for
            time t given observations from times [0...t-1]

        Returns
        -------
        smoothed_state_means : [n_timesteps, n_dim_state]
            mean of hidden state distributions for times [0...n_timesteps-1] given
            all observations
        smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            covariance matrix of hidden state distributions for times
            [0...n_timesteps-1] given all observations
        """

        n_timesteps = len(self.filtered_state_means)

        smoothed_state_means = np.zeros((n_timesteps, self.nstate))
        smoothed_state_covariances = np.zeros((n_timesteps, self.nstate,
                                                    self.nstate))

        kalman_smoothing_gains = np.zeros((n_timesteps - 1, self.nstate,
                                           self.nstate))

        smoothed_state_means[-1] = self.filtered_state_means[-1]
        smoothed_state_covariances[-1] = self.filter_state_covariances[-1]

        for t in reversed(range(n_timesteps - 1)):
            try:
                psc_inv = np.linalg.pinv(
                               self.predicted_state_covariances[t+1])
            except:
                psc_inv = np.linalg.inv(
                               self.predicted_state_covariances[t+1])
            kalman_smoothing_gains[t] = (
                np.dot(self.filtered_state_covariances[t],
                       np.dot(self.transition_matrix.T, psc_inv))
                )

            smoothed_state_means[t] = (self.filtered_state_means[t]
                + np.dot(kalman_smoothing_gains[t],
                         (smoothed_state_means[t+1]
                          - self.predicted_state_means[t+1]
                          )
                         )
            )
            smoothed_state_covariances[t] = (
                self.filtered_state_covariances[t]
                + np.dot(kalman_smoothing_gains[t],
                         np.dot((smoothed_state_covariances[t+1]
                                 - self.predicted_state_covariances[t+1]),
                                kalman_smoothing_gains[t].T
                                )
                         )
            )

        return smoothed_state_means, smoothed_state_covariances

    def initialize(self, oseries):
        """Initialize sequential processing of the Kalman filter by
        constructing observation matrices allowing missing values.

        Parameters
        ----------
        oseries : pandas.DataFrame
            multiple time series

        Returns
        -------
        None
        """
        self.oseries_index = oseries.index
        observations_masked = np.ma.array(oseries,
                                          mask=(~np.isfinite(oseries)))
        (n_timesteps, dimobs) = observations_masked.shape
        self.observation_indices = np.zeros((n_timesteps, dimobs), dtype=np.float64)
        self.observation_count = np.zeros(n_timesteps, dtype=np.int64)
        self.observations = np.zeros((n_timesteps, dimobs), dtype=np.float64)

        for t in range(n_timesteps):
            observation = observations_masked[t]
            # add large value to find all finite non-masked values
            obstmp = observation + 1e10
            obsindices = obstmp.nonzero()[0]
            self.observation_count[t] = len(obsindices)

            if (len(obsindices) > 0):
                for i in range(len(obsindices)):
                    obsid = int(obsindices[i])
                    self.observations[t, obsid] = observation[obsid]
                    self.observation_indices[t, i] = obsid

    def run_filter(self, initial_state_mean=None,
                   initial_state_covariance=None,
                   warmup=1):
        """
        Apply the Kalman Filter at time :math:`t = [0,...,N]`
        given observations up to and including time `t`.

        This is a sequential processing implementation of the Kalman filter
        requiring a diagonal observation error covariance matrix.
        The algorithm allows for missing data using the arrays
        observation_count giving the number of observations for each timestep,
        and observation_indices containing the corresponding indices
        of those observations used to select the appropriate rows
        from observation_matrix and observation_variance.
        These arrays have been constructed with self.initialize()

        Parameters
        ----------
        initial_state_mean : array [n_dim_state]
            state vector for initializing Kalman filter
        initial_state_covariance : array [n_dim_state, n_dim_state]
            state covariance matrix for initializing Kalman filter

        Returns
        -------
        sigmas : list [n_timesteps]
            scaling variance :math:`\\sigma_{*}^{2}=\\nu_{t}F_{t}\\nu_{t}^{T}`
        detfs : list [n_timesteps]
            determinant of innovation covariance :math:`|F_{t}|`
        nobs : list [n_timesteps]
            number of observations for each time step

        """

        if initial_state_mean is None:
            initial_state_mean = np.zeros(self.nstate)
        if initial_state_covariance is None:
            initial_state_covariance = np.eye(self.nstate)

        # Kalman filter
        (sigmas, detfs, sigmacount,
         filtered_state_means,
         filtered_state_covariances) = \
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
