"""
Created on 28/05/20
@author: Talip Turkmen
"""

from utilities.partitioning import partition


class Subcampaign:

    def __init__(self, classes, slots):
        self.classes = classes
        self.slots = slots
        self.number_of_interested_users = []

    def get_slot(self, id):
        if id < len(self.slots):
            return self.slots[id]
        return False

    def sample(self, save_sample=True):
        self.number_of_interested_users = tuple([c.sample(save_sample) for c in self.classes])
        return self.number_of_interested_users

    def get_classes_ids(self):
        return tuple([c.id for c in self.classes])

    def get_class_names(self):
        return tuple([c.name for c in self.classes])

    ## TODO WILL CHANGE
    def disaggregate(self):
        class_indices = list(range(len(self.classes)))
        all_partitions = partition(class_indices)
        subcampaigns = []
        for p in all_partitions:
            p_subcampaigns = []
            for indices in p:
                p_subcampaigns.append(Subcampaign([self.classes[i].copy() for i in indices]))
            subcampaigns.append(p_subcampaigns)
        return subcampaigns

    def get_samples(self):
        samples = []
        for t in range(len(self.classes[0].samples)):
            iteration_samples = [c.samples[t] for c in self.classes]
            samples.append((sum([x[0] for x in iteration_samples]), tuple(map(lambda s: s[1], iteration_samples))))
        return samples

    def copy(self):
        return Subcampaign([c.copy() for c in self.classes])
