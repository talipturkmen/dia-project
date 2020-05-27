from environment.environment import Environment
from environment.click_function import Click_Function
from environment.subcampaign import Subcampaign

from distributions.gaussian import Gaussian


def build_environment(configuration):
    subcampaigns = []
    click_function_id = 0

    # LOOP OVER SUBCAMPAIGNS
    for subcampaign_data in configuration:
        click_functions = []

        #  LOOP OVER CLASSES
        for classes in subcampaign_data:
            noise = Gaussian(classes['noise_mean'], classes['noise_variance'])
            click_probs = classes['click_probabilities']
            click_function = Click_Function(noise, classes['max_value'],
                                            classes['offset'], classes['speed'],click_probs,
                                            click_function_id)
            click_function_id += 1
            click_functions.append(click_function)

        subcampaigns.append(Subcampaign(click_functions))

    return Environment(subcampaigns)
