# -*- coding: utf-8 -*-
"""Created on Mon Mar 29 08:47:25 2021.

@author: Wilbert Berendrecht
"""

import numpy as np

from KalmanFilterf95 import kfseq


class SPKalmanFilter():

    def __init__(self):
        """
        transition_matrix : [n_dim_state, n_dim_state] array
            state transition matrix from time t-1 to t
        transition_covariance : [n_dim_state, n_dim_state] array
            covariance matrix for state transition from time t-1 to t
        observation_matrix : [n_dim_obs, n_dim_state] array
            observation matrix for time t
        observation_variance : [n_dim_obs] array
            diagonal of covariance matrix for observation at time t
        """

        self.filtered_state_means = None
        self.filtered_state_covariances = None
        self.predicted_state_means = None
        self.predicted_state_covariances = None
        self.detfs = None
        self.sigmas = None
        self.nobs = None

    def set_matrices(self, transition_matrix, transition_covariance,
                     observation_matrix, observation_variance):
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        self.observation_matrix = observation_matrix
        self.observation_variance = observation_variance
        self.nstate = self.transition_matrix.shape[0]

    def get_mle(self):
        return (self.nobs * np.log(2*np.pi) + np.sum(self.detfs)
                + np.sum(self.sigmas)
                )

    def get_scale(self):
        return np.sum(self.sigmas) / self.nobs

    def filter_predict(self, current_state_mean, current_state_covariance):
        """Calculate the mean and covariance of :math:`P(x_{t|t-1})`

        Using the mean and covariance of :math:`P(x_{t-1|t-1})`,
        calculate the mean and covariance of :math:`P(x_{t|t-1})`.

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

        predicted_state_mean = (
            np.dot(self.transition_matrix, current_state_mean)
        )
        predicted_state_covariance = (
            np.dot(self.transition_matrix,
                   np.dot(current_state_covariance, self.transition_matrix.T))
            + self.transition_covariance
        )

        return (predicted_state_mean, predicted_state_covariance)

    def filter_update(self, t, corrected_state_mean,
                      corrected_state_covariance):
        """Update a predicted state with a Kalman Filter update using
        sequential processing (assumption of uncorrelated observation error)

        Incorporate observation `observation` from time `t` to turn
        :math:`P(x_{t|t-1})` into :math:`P(x_{t|t})` and to obtain
        `sigma` and :math:`|F_t|)`

        Parameters
        ----------
        corrected_state_mean : [n_dim_state] array
            mean of state at time t given observations from times
            [0...t-1], i.e. predicted state mean
        corrected_state_covariance : [n_dim_state, n_dim_state] array
            covariance of state at time t given observations from times
            [0...t-1], i.e. predicted state covariance
        observation : [n_dim_obs] array
            observation at time t, where the observations used for updating
            are defined by `observation_indices`
        nobs : integer
            number of actual observations at time t (without missing values)

        Returns
        -------
        corrected_state_mean : [n_dim_state] array
            mean of state at time t given observations from times
            [0...t], i.e. updated state mean
        corrected_state_covariance : [n_dim_state, n_dim_state] array
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
        observation = self.observations[t, :]
        n_observation = np.int(self.observation_count[t])
        observation_indices = self.observation_indices[t, :]

        for i in range(n_observation):
            observation_index = observation_indices[i]
            obsmat = self.observation_matrix[observation_index, :]

            innovation = (observation[observation_index]
                          - np.dot(obsmat, corrected_state_mean))

            dot_statecov_obsmat = np.dot(corrected_state_covariance, obsmat)
            innovation_covariance = (np.dot(obsmat, dot_statecov_obsmat)
                                     + self.observation_variance[observation_index])

            kgain = dot_statecov_obsmat / innovation_covariance

            corrected_state_covariance = (corrected_state_covariance
                                          - (np.outer(kgain, kgain)
                                             * innovation_covariance))

            corrected_state_mean = corrected_state_mean + kgain * innovation

            sigma = sigma + (innovation**2 / innovation_covariance)
            detf = detf + np.log(innovation_covariance)

        return (corrected_state_mean,
                corrected_state_covariance,
                sigma, detf)

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
        observations_masked = np.ma.array(oseries,
                                          mask=(~np.isfinite(oseries)))
        (n_timesteps, dimobs) = observations_masked.shape
        self.observation_indices = np.zeros((n_timesteps, dimobs), dtype=int)
        self.observation_count = np.zeros(n_timesteps, dtype=int)
        self.observations = np.zeros((n_timesteps, dimobs))

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

    def run(self, initial_state_mean=None, initial_state_covariance=None,
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
        # get number of time steps as number of Kalman filter recursions
        n_timesteps = self.observations.shape[0]

        # initialize parameters for concentrated loglikelihood
        sigmas = []
        detfs = []
        sigmacount = []
        if initial_state_mean is None:
            initial_state_mean = np.zeros(self.nstate)
        else:
            initial_state_mean = initial_state_mean
        if initial_state_covariance is None:
            initial_state_covariance = np.eye(self.nstate)
        else:
            initial_state_covariance = initial_state_covariance

        # initialize Kalman filter states and covariances
        self.filtered_state_means = []
        self.filtered_state_covariances = []
        self.predicted_state_means = []
        self.predicted_state_covariances = []

        filtered_state_mean = initial_state_mean
        filtered_state_covariance = initial_state_covariance

        for t in range(n_timesteps):
            # Kalman filter prediction step
            predicted_state_mean, predicted_state_covariance = (
                self.filter_predict(filtered_state_mean,
                                    filtered_state_covariance)
            )
            self.predicted_state_means.append(predicted_state_mean)
            self.predicted_state_covariances.append(predicted_state_covariance)

            n_observation = np.int(self.observation_count[t])
            if n_observation > 0:
                # Kalman filter update step
                (filtered_state_mean,
                 filtered_state_covariance, sigma, detf) = (
                    self.filter_update(t,
                                       predicted_state_mean, predicted_state_covariance
                                       )
                )
                # construct list of values used for concentrated loglikelihood
                sigmas.append(sigma)
                detfs.append(detf)
                sigmacount.append(n_observation)
            else:
                filtered_state_mean = predicted_state_mean
                filtered_state_covariance = predicted_state_covariance

            self.filtered_state_means.append(filtered_state_mean)
            self.filtered_state_covariances.append(filtered_state_covariance)

        self.detfs = detfs[warmup:]
        self.sigmas = sigmas[warmup:]
        self.nobs = np.sum(self.observation_count[warmup:])

    def runf95(self, initial_state_mean=None, initial_state_covariance=None,
               warmup=1):
        if initial_state_mean is None:
            initial_state_mean = np.zeros(self.nstate)
        else:
            initial_state_mean = initial_state_mean
        if initial_state_covariance is None:
            initial_state_covariance = np.eye(self.nstate)
        else:
            initial_state_covariance = initial_state_covariance

        sigmas, detfs, sigmacount = kfseq(self.observations,
                                          self.transition_matrix,
                                          self.transition_covariance,
                                          self.observation_matrix,
                                          self.observation_variance,
                                          self.observation_indices,
                                          self.observation_count,
                                          initial_state_mean,
                                          initial_state_covariance)

        self.detfs = detfs[warmup:sigmacount]
        self.sigmas = sigmas[warmup:sigmacount]
        self.nobs = np.sum(self.observation_count[warmup:])
