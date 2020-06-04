from algorithms.knapsack import Knapsack
from algorithms.hungarian_algorithm import hungarian_algorithm
from utilities.environment_building import build_environment
from config.config import get_configuration
from utilities.plot_function import plot_function

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    timesteps = 250
    daily_budget = 100
    budget_discretization_density = 20
    budget_discretization_steps = [i * daily_budget / budget_discretization_density for i in
                                   range(budget_discretization_density + 1)]

    plot_path = "plots/"

    ### Build environment ###
    environment = build_environment(get_configuration())

    num_subcampaigns = len(environment.subcampaigns)

    ### Plot environment data ###
    legend = []
    for subcampaign in environment.subcampaigns:
        plt.figure()
        number_of_interested_users = subcampaign.sample(save_sample=False)
        for slot in subcampaign.slots:
            plot_function(slot.get_real, range(daily_budget))
            plot_function(lambda x: sum(slot.sample_for_all_classes(x, number_of_interested_users, save_sample=False)),
                          range(daily_budget))
            legend.append("Subcampaign: " + str(subcampaign.name) + " Slot: " + str(slot.id) + " real click function")
            legend.append("Subcampaign: " + str(subcampaign.name) + " Slot: " + str(slot.id) + " noisy click function")

        plt.legend(legend, bbox_to_anchor=(1.05, 1), loc=2)
        plt.title("Environment - Aggregated")
        plt.savefig(plot_path + 'environment_aggregated-' + str(subcampaign.name) +
                    '.png', bbox_inches='tight', dpi=300)
        legend.clear()
        plt.close()

    exit(0)
