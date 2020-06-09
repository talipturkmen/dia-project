"""
Created on 28/05/20
@author: Talip Turkmen
"""
import numpy as np


class CampaignClass:

    def __init__(self, name, id, mean, variance, sigma, max_value, offset, speed, prominence_rates):
        self.name = name
        self.id = id
        self.mean = mean
        self.variance = variance
        self.sigma = sigma
        self.max_value = max_value
        self.offset = offset
        self.speed = speed

        # TODO: This loop will be beautified
        for i, p in enumerate(prominence_rates):
            for j in range(i):
                prominence_rates[i] = p * prominence_rates[j]

        self.prominence_rates = np.array(prominence_rates)
        self.interested_users = []
        self.clicks = []

    def sample_interested_users(self, save_sample=True):
        sample = np.random.normal(self.mean, self.variance)
        if save_sample:
            self.interested_users.append(sample)
        return sample

    def sample_clicks(self, x, save_sample=True):
        """

        :param x: bidding
        :param save_sample:
        :return: expected number of clicks according to bid
        """
        number_of_click = min(np.random.normal(self.real_function_value(x), self.sigma), self.interested_users[-1])
        number_of_clicks = number_of_click * self.prominence_rates

        if save_sample:
            self.clicks.append(number_of_clicks)

        return number_of_clicks

    def real_function_value(self, x):
        """
        :param x: bid
        :return: number of clicks without noise
        """

        return max(0, (self.max_value - (self.offset * np.exp(-self.speed * x))))

    def real_function_value_for_slots(self, x):
        """
        :param x: bid
        :return: number of clicks without noise
        """
        return max(0, (self.max_value - (self.offset * np.exp(-self.speed * x)))) * self.prominence_rates

    def copy(self):
        cf = CampaignClass(self.name, self.id, self.mean, self.variance, self.sigma, self.max_value, self.offset,
                           self.speed, self.prominence_rates)
        cf.interested_users = self.interested_users.copy()
        cf.clicks = self.clicks.copy()
        return cf
