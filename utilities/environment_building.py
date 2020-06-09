"""
Created on 28/05/20
@author: Talip Turkmen
"""

import numpy as np

from configuration.config import get_configuration
from environment.classes import CampaignClass
from environment.environment import Environment
from environment.subcampaign import Subcampaign
from experiment.subcampaign_algo import SlotAlgorithm


def build_environment(configuration):
    subcampaigns = []
    # LOOP OVER SUBCAMPAIGNS
    for ii, subcampaign_data in enumerate(configuration):
        classes = []

        #  LOOP OVER CLASSES
        class_id = 0
        for class_ in subcampaign_data:
            user_class = CampaignClass(name=class_['class_name'],
                                       id=class_id,
                                       mean=class_['mean'],
                                       variance=class_['variance'],
                                       sigma=class_['sigma'],
                                       max_value=class_['max_value'],
                                       offset=class_['offset'],
                                       speed=class_['speed'],
                                       prominence_rates=np.array(class_['slots_prominence_rates']))

            classes.append(user_class)
            class_id += 1

        subcampaigns.append(Subcampaign(ii, classes))

    return Environment(subcampaigns)


if __name__ == '__main__':
    # FOR TESTING PURPOSES
    # PLEASE SEE THE USAGE BELOW FOR SAMPLING

    env = build_environment(get_configuration())
    subcampaign0 = env.get_subcampaign(0)
    interested_users = subcampaign0.sample_interested_users()

    print('Expected interested users per class for subcampaign0', interested_users)

    expected_clicks_for_subcampaign0_class1 = subcampaign0.sample_clicks(x=80, class_id=1)
    print('Expected number of clicks for the subcampaign0 wrt slots:', expected_clicks_for_subcampaign0_class1)
    print('Expected number of clicks for the subcampaign0 slot:0:', expected_clicks_for_subcampaign0_class1[0])

    expected_click_for_all_classes_of_subcampaign0 = subcampaign0.sample_clicks_for_all_classes_agg(x=80)
    print(
        'Expected number of clicks for the subcampaign:0 and all classes with bid:80 ',
        expected_click_for_all_classes_of_subcampaign0)
    print(
        'Expected number of clicks for the subcampaign:0 class:aggreagated slot:0 and all classes with bid:80 ',
        expected_click_for_all_classes_of_subcampaign0[0])

    daily_budget = 100
    budget_discretization_density = 20
    budget_discretization_steps = [i * daily_budget / budget_discretization_density for i in
                                   range(budget_discretization_density + 1)]
    GPTS_prior = lambda x: 3 * x

    slot_algos = []
    slot_dict = []
    for subcampaign in env.subcampaigns:
        subcampaign.sample_interested_users()
        samples = [[k * daily_budget / 50 for k in range(51)],
                   [subcampaign.sample_clicks_for_all_classes_agg(k * daily_budget / 50, save_sample=False)
                    for k in range(51)]]

        for slot_id in range(len(subcampaign.classes[0].prominence_rates)):
            slot_algo = SlotAlgorithm(subcampaign.name, slot_id, budget_discretization_steps.copy(), GPTS_prior)
            slot_samples = [samples[0], np.array(samples[1])[:, slot_id]]
            slot_algo.learn_gp_kernel_hyperparameters(slot_samples)
            slot_algos.append(slot_algo)
            slot_dict.append({'subcampaign': subcampaign.name,
                              'slot': slot_id
                              })

    exit(0)
