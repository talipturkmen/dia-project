"""
Created on 28/05/20
@author: Talip Turkmen
"""
import numpy as np


class Class(object):
    def __init__(self, name, id):
        self.name = name
        self.id = id


class CampaignClass(Class):

    def __init__(self, name, id, mean, sigma):
        super().__init__(name, id)
        self.mean = mean
        self.sigma = sigma
        self.samples = []

    def sample(self, save_sample=True):
        sample = np.random.normal(self.mean, self.sigma)
        if save_sample:
            self.samples.append(sample)
        return sample


class ClickClass(Class):
    def __init__(self, name, id, slot_id, sigma, max_value, offset, speed):
        super().__init__(name, id)
        self.sigma = sigma
        self.slot_id = slot_id
        self.max_value = max_value
        self.offset = offset
        self.speed = speed
        self.samples = []

    def sample(self, x, interested_users, save_sample=True):
        '''

        :param x: bidding
        :param interested_users: number of interested users
        :param save_sample:
        :return: expected number of clicks according to bid
        '''

        number_of_clicks = min(np.random.normal(self.real_function_value(x), self.sigma), interested_users)
        if save_sample:
            self.samples.append(number_of_clicks)

        return number_of_clicks

    def real_function_value(self, x):
        '''

        :param x: bid
        :return: number of clicks without noise
        '''
        return max(0, (self.max_value - (self.offset * np.exp(-self.speed * x))))
