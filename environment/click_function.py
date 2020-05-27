import numpy as np

from distributions.gaussian import Gaussian


class Click_Function:

    # noise : Probability_Distribution
    def __init__(self, noise, max_height, offset, speed, slot_click_probs, id, calculateOffset=True):
        self.noise = noise
        self.max_height = max_height
        if calculateOffset:
            self.offset = max_height / np.exp(-speed * offset)
        else:
            self.offset = offset
        self.speed = speed
        self.samples = []
        self.slot_click_probs = slot_click_probs
        self.id = id

    def sample(self, x, save_sample=True):
        ## TODO: JUST SAMPLE FROM GAUSSIAN
        interested_users = self.real_function_value(x) + self.noise.sample()  # NUMBER OF INTERESTED USERS
        slot_clicks = []
        for click_prob in self.slot_click_probs:
            prob = Gaussian(click_prob['mean'], click_prob['variance'])
            slot_click_count = interested_users * prob
            slot_clicks.append(slot_click_count)
        if save_sample:
            self.samples.append((x, interested_users, slot_clicks))
        return slot_clicks



## TODO: WHY USED?
def real_function_value(self, x):
    return max(0, (self.max_height - (self.offset * np.exp(-self.speed * x))))


def copy(self):
    cf = Click_Function(self.noise, self.max_height, self.offset, self.speed, self.id, calculateOffset=False)
    cf.samples = self.samples.copy()
    return cf
