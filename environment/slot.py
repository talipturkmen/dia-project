"""
Created on 28/05/20
@author: Talip Turkmen
"""


class Slot:
    def __init__(self, id, classes):
        self.id = id
        self.classes = classes
        self.samples = []

    def sample_for_all_classes(self, x, interested_users, save_sample=True):
        '''

        :param x: bid
        :param type: list[] interested_users: number of interested users for subcampign and all class
        :param save_sample:
        :return: returns expected number of clicks for the slot for each class given the bid
        '''
        self.samples = tuple([c.sample(x, interested_users[i], save_sample) for i, c in enumerate(self.classes)])
        return self.samples

    def sample(self, x, interested_users, class_id, save_sample=True):
        '''

        :param x: bid
        :param interested_users: number of interested users for subcampign and class
        :param class_id:
        :param save_sample:
        :return: returns expected number of clicks for the slot given class and the bid
        '''
        self.samples = self.classes[class_id].sample(x, interested_users, save_sample)
        return self.samples

    def get_clicks(self, x, interested_users):
        clicks = []
        for c in self.classes:
            c.sample(x, interested_users)

        return clicks

    def previous_clicks(self, class_id):
        return self.classes[class_id].samples

    def get_real(self, x):
        val = 0
        for c in self.classes:
            val = val + c.real_function_value(x / len(self.classes))
        return val
