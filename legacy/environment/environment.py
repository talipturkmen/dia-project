"""
Created on 28/05/20
@author: Talip Turkmen
"""


class Environment:
    def __init__(self, subcampaigns):
        self.subcampaigns = subcampaigns

    def get_subcampaign(self, id):
        if id < len(self.subcampaigns):
            return self.subcampaigns[id]
        return False

    ## TODO WILL CHANGE
    def get_combinations(self):
        result = [[]]
        for subcampaign in self.subcampaigns:
            splits = subcampaign.disaggregate()
            new_result = []
            for combination in result:
                for split in splits:
                    new_result.append([s.copy() for s in combination] + [s.copy() for s in split])
            result = new_result
        return result

    def copy(self):
        return Environment([s.copy() for s in self.subcampaigns])

    def get_number_of_all_slots(self):
        num_slots = 0
        for subcampaign in self.subcampaigns:
            num_slots += len(subcampaign.classes[0].prominence_rates)

        return num_slots
