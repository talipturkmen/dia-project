"""
Created on 28/05/20
@author: Talip Turkmen
"""

import numpy as np


class Subcampaign:

    def __init__(self, name, classes):
        self.name = name
        self.classes = classes
        self.slot_num = len(classes[0].prominence_rates)
        self.number_of_interested_users = []

    def sample_interested_users(self, save_sample=True):
        self.number_of_interested_users = tuple([c.sample_interested_users(save_sample) for c in self.classes])
        return self.number_of_interested_users

    def sample_clicks_for_all_classes_agg(self, x, save_sample=True):
        """
        :param x: bid
        :param save_sample:
        :return: returns expected number of clicks for the slot for each class given the bid
        """

        self.clicks = tuple([c.sample_clicks(x / len(self.classes), save_sample) for c in self.classes])

        return np.sum(np.array(self.clicks), axis=0)

    def sample_clicks_for_all_classes_disagg(self, x, save_sample=True):
        """
        :param x: bid
        :param save_sample:
        :return: returns expected number of clicks for the slot for each class given the bid
        """

        self.clicks = tuple([c.sample_clicks(x, save_sample) for c in self.classes])

        return self.clicks

    def sample_clicks(self, x, class_id, save_sample=True):
        """
        :param x: bid
        :param class_id:
        :param save_sample:
        :return: returns expected number of clicks for the slot given class and the bid
        """
        return self.classes[class_id].sample_clicks(x, save_sample=save_sample)

    def get_classes_ids(self):
        return tuple([c.id for c in self.classes])

    def get_class_names(self):
        return tuple([c.name for c in self.classes])

    def get_samples(self):
        samples = []
        for t in range(len(self.classes[0].samples)):
            iteration_samples = [c.samples[t] for c in self.classes]
            samples.append((sum([x[0] for x in iteration_samples]), tuple(map(lambda s: s[1], iteration_samples))))
        return samples

    def copy(self):
        return Subcampaign(self.name, [c.copy() for c in self.classes])

    def get_real_agg(self, x):
        return np.sum(np.array(tuple([c.real_function_value_for_slots(x / len(self.classes)) for c in self.classes])),
                      axis=0)

    def get_real_disagg(self, x):
        return tuple([c.real_function_value_for_slots(x) for c in self.classes])
