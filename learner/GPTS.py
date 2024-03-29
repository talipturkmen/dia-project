"""
Created on 09/06/20
@author: Talip Turkmen
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS:

    def __init__(self, arms, kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)), alpha=10.0, prior=None):
        self.n_arms = len(arms)
        self.predicted_arms = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.arms = arms
        self.collected_rewards = np.array([])
        self.pulled_arms = []
        self.gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True,
                                                         n_restarts_optimizer=9)
        if prior is None:
            self.prior = lambda x: 0
        else:
            self.prior = prior

    # Fit the gaussian process using the pulled arms and their rewards.
    # It saves the results of the prediction in self.predicted_arms and saves the sigma of each arm in self.sigmas

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gaussian_process.fit(x, y)
        self.means, self.sigmas = self.gaussian_process.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-4)  # sigmas must be positive
        self.means += np.array([self.prior(arm) for arm in self.arms])

        sigma_bonus = np.array([0 / (len(list(filter(lambda x: x == arm, self.pulled_arms))) + 1) for arm in self.arms])
        self.predicted_arms = np.random.normal(self.means, self.sigmas + sigma_bonus)

        self.predicted_arms = np.maximum(0, self.predicted_arms)  # predictions must be nonnegative (for the knapsack)

    # Add the new observation in the model
    def update_observations(self, arm_idx, reward):
        self.update_observations_raw(self.arms[arm_idx], reward)

    # Add the new observation in the model (pass directly the arm value, not the arm index)
    # x can be any point, not only an arm value
    def update_observations_raw(self, x, reward):
        self.pulled_arms.append(x)
        self.collected_rewards = np.append(self.collected_rewards, reward - self.prior(x))

    # Get the prediction of a given arm
    def sample(self, arm_idx):
        return self.predicted_arms[arm_idx]

    # Get the lower bound of a given arm
    def lower_bound(self, arm_idx):
        n_samples = len(list(filter(lambda x: x == self.arms[arm_idx], self.pulled_arms)))
        if n_samples == 0:
            return -np.inf
        else:
            return self.means[arm_idx] - 1.96 * self.sigmas[arm_idx] / np.sqrt(n_samples)

    # Get the variance of a given arm
    def get_sigma(self, arm_idx):
        return self.sigmas[arm_idx]

    # Find the index of the given arm in self.arm
    def find_arm(self, arm):
        for idx in range(self.n_arms):
            if self.arms[idx] == arm:
                return idx
        return False

    def learn_kernel_hyperparameters(self, samples):
        x = np.atleast_2d(samples[0]).T
        y = [y - self.prior(x) for (x, y) in zip(samples[0], samples[1])]
        self.gaussian_process.fit(x, y)
        self.gaussian_process = GaussianProcessRegressor(kernel=self.gaussian_process.kernel_,
                                                         alpha=self.gaussian_process.alpha,
                                                         normalize_y=True, n_restarts_optimizer=0)

    # Get the regression error (MSE) with respect to the training samples for a given arm
    def get_arm_average_regression_error(self, arm, points_to_evaluate=None):
        total_squared_error = 0
        count = 0

        if points_to_evaluate is None:
            points_to_evaluate = zip(self.pulled_arms, self.collected_rewards)

        for (pulled_arm, collected_reward) in points_to_evaluate:
            if pulled_arm == arm:
                count += 1
                x = self.means[self.find_arm(pulled_arm)]
                total_squared_error += (x - self.prior(pulled_arm) - collected_reward) ** 2

        if count == 0:
            return -1
        return np.sqrt(total_squared_error) / count

    def get_average_regression_error(self, points_to_evaluate=None):
        total_squared_error = 0

        if points_to_evaluate is None:
            points_to_evaluate = zip(self.pulled_arms, self.collected_rewards)

        for (pulled_arm, collected_reward) in points_to_evaluate:
            x = self.means[self.find_arm(pulled_arm)]
            total_squared_error += (x - self.prior(pulled_arm) - collected_reward) ** 2
        return np.sqrt(total_squared_error) / len(points_to_evaluate)
