"""
Created on 28/05/20
@author: Talip Turkmen
"""

from environment.environment import Environment
from environment.subcampaign import Subcampaign
from environment.slot import Slot
from environment.classes import CampaignClass, ClickClass
from config.config import get_configuration


def build_environment(configuration):
    subcampaigns = []
    # LOOP OVER SUBCAMPAIGNS
    for ii, subcampaign_data in enumerate(configuration):
        classes_ = []
        slots = []
        slot_classes = []

        #  LOOP OVER CLASSES
        class_id = 0
        for classes in subcampaign_data:
            user_class = CampaignClass(classes['class_name'], class_id, mean=classes['mean'], sigma=classes['variance'])
            # LOOP OVER SLOTS
            slot_id = 0
            for slot in classes['click_probabilities']:
                slot_class = ClickClass(user_class.name,
                                        user_class.id,
                                        slot_id=slot_id,
                                        sigma=slot['sigma'],
                                        max_value=slot['max_value'],
                                        offset=slot['offset'],
                                        speed=slot['speed'])
                slot_classes.append(slot_class)
                slot_id += 1

            classes_.append(user_class)
            class_id += 1

        for i in range(slot_id):
            classes = []
            for j in range(len(slot_classes)):
                if j % slot_id == i:
                    classes.append(slot_classes[j])

            slots.append(Slot(i, classes))

        subcampaigns.append(Subcampaign(ii, classes_, slots))

    return Environment(subcampaigns)

    if __name__ == '__main__':
        ### FOR TESTING PURPOSES ####
        ### PLEASE SEE THE USAGE BELOW FOR SAMPLING ####
        env = build_environment(get_configuration())
        subcampaign0 = env.get_subcampaign(0)
        interested_users = subcampaign0.sample()

        print('Expected interested users per class for subcampaign0', interested_users)

        slot0 = subcampaign0.get_slot(0)
        expected_click_for_slot0_and_class2 = slot0.sample(x=80, interested_users=interested_users[1], class_id=1)
        print(
            'Expected number of clicks for the subcampaign:0 and the slot:0 and class 2 with bid:80',
            expected_click_for_slot0_and_class2)

        expexted_click_for_all_classes_and_slot0 = slot0.sample_for_all_classes(x=80,
                                                                                interested_users=interested_users, )
        print(
            'Expected number of clicks for the subcampaign:0 and the slot:0 and all classes with bid:80 ',
            expexted_click_for_all_classes_and_slot0)
        exit(0)
