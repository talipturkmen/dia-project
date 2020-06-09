"""
Created on 09/06/20
@author: Talip Turkmen
"""

import matplotlib.pyplot as plt

from configuration.config import get_configuration
from experiment.experiment import *
from utilities.environment_building import build_environment

if __name__ == '__main__':

    timesteps_stationary = 250
    timesteps_context_generation = 250

    daily_budget = 100
    budget_discretization_density = 20
    budget_discretization_steps = [i * daily_budget / budget_discretization_density for i in
                                   range(budget_discretization_density + 1)]

    plot_path = "plots/"

    # Build environment
    environment = build_environment(get_configuration())

    num_subcampaigns = len(environment.subcampaigns)

    # Plot environment data
    legend = []
    for subcampaign in environment.subcampaigns:
        plt.figure()
        subcampaign.sample_interested_users()
        real_values = []
        sample_values = []
        for x in range(daily_budget):
            real_values.append(subcampaign.get_real_agg(x))
            sample_values.append(subcampaign.sample_clicks_for_all_classes_agg(x))

        for slot_id in range(subcampaign.slot_num):
            plt.plot(range(daily_budget), np.array(real_values)[:, slot_id])
            plt.plot(range(daily_budget), np.array(sample_values)[:, slot_id])

            legend.append("Slot: " + str(slot_id) + " real click function")
            legend.append("Slot: " + str(slot_id) + " noisy click function")

        plt.legend(legend, bbox_to_anchor=(1.05, 1), loc=2)
        plt.title("Environment - Aggregated Subcampaign-" + str(subcampaign.name))
        plt.savefig(plot_path + 'environment_aggregated-' + str(subcampaign.name) +
                    '.png', bbox_inches='tight', dpi=300)
        legend.clear()
        plt.close()

    legend = []

    for subcampaign in environment.subcampaigns:
        plt.figure()
        real_values = []
        sample_values = []
        for x in range(daily_budget):
            real_values.append(subcampaign.get_real_disagg(x))
            sample_values.append(subcampaign.sample_clicks_for_all_classes_disagg(x))
        for c in subcampaign.classes:
            class_real_values = np.array(real_values)[:, c.id, :]
            class_sample_values = np.array(sample_values)[:, c.id, :]

            for slot_id in range(subcampaign.slot_num):
                plt.plot(range(daily_budget), class_real_values[:, slot_id])
                plt.plot(range(daily_budget), class_sample_values[:, slot_id])

                legend.append("Class:" + str(c.id) + " Slot: " + str(slot_id) + " real click function")
                legend.append("Class:" + str(c.id) + " Slot: " + str(slot_id) + " noisy click function")

        plt.legend(legend, bbox_to_anchor=(1.05, 1), loc=2)
        plt.title("Environment - Disaggregated Subcampaign-" + str(subcampaign.name))
        plt.savefig(plot_path + 'environment_disaggregated-' + str(subcampaign.name) +
                    '.png', bbox_inches='tight', dpi=300)
        legend.clear()
        plt.close()

    ############################################
    ## Build the clairvoyant solution
    ############################################
    # With aggregated subcampaigns
    real_values = []
    slot_values = []
    slot_dict = []
    for subcampaign in environment.subcampaigns:
        real_values = [subcampaign.get_real_agg(arm) for arm in budget_discretization_steps]
        for slot_id in range(subcampaign.slot_num):
            slot_value = np.array(real_values)[:, slot_id]
            slot_values.append(slot_value)
            slot_dict.append({'subcampaign': subcampaign.name,
                              'slot': slot_id
                              })

    optimal_super_arm = Knapsack(daily_budget, slot_values).optimize()
    print("Optimal superarm is " + prettify_super_arm(environment, optimal_super_arm, slot_dict))
    optimal_super_arm_value = 0
    for (i, arm) in optimal_super_arm:
        subcampaign_values = environment.subcampaigns[slot_dict[i]['subcampaign']].get_real_agg(arm)
        optimal_super_arm_value += np.array(subcampaign_values)[slot_dict[i]['slot']]

    print("Value of optimal superarm = " + str(optimal_super_arm_value))
    # TODO: There is a problem with the real function value. It must be defined correctly

    # With fully disaggregated subcampaigns
    real_values = []
    slot_values = []
    slot_dict = []
    subclass_index = 0
    for subcampaign in environment.subcampaigns:

        real_values = [subcampaign.get_real_disagg(arm) for arm in budget_discretization_steps]
        for subclass in subcampaign.classes:
            for slot_id in range(subcampaign.slot_num):
                slot_value = np.array(real_values)[:, subclass.id, slot_id]
                slot_values.append(slot_value)
                slot_dict.append({'subcampaign': subcampaign.name,
                                  'subclass': subclass.id,
                                  'slot': slot_id
                                  })
    optimal_disaggregated_super_arm = Knapsack(daily_budget, slot_values).optimize()
    print("Optimal disaggregated superarm is " + str(
        [(slot_dict[i]['subcampaign'], slot_dict[i]['subclass'], slot_dict[i]['slot'], arm) for (i, arm) in
         optimal_disaggregated_super_arm]))

    optimal_disaggregated_super_arm_value = 0
    for (i, arm) in optimal_disaggregated_super_arm:
        subcampaign_values = environment.subcampaigns[slot_dict[i]['subcampaign']].get_real_disagg(arm)
        optimal_disaggregated_super_arm_value += np.array(subcampaign_values)[slot_dict[i]['subclass'],
                                                                              slot_dict[i]['slot']]

    print("Value of optimal disaggregated superarm = " + str(optimal_disaggregated_super_arm_value))

    clairvoyant_rewards = [optimal_super_arm_value for _ in range(timesteps_stationary)]
    disaggregated_clairvoyant_rewards = [optimal_disaggregated_super_arm_value for _ in
                                         range(timesteps_context_generation)]

    ############################################
    # Define GPTS prior
    ############################################

    GPTS_prior = lambda x: 3 * x

    ############################################
    # Perform experiments
    ############################################

    experiment = Experiment(environment, budget_discretization_steps, daily_budget, GPTS_prior)

    print("------ GPTS stationary ------")
    (stationary_rewards, stationary_final_environment, stationary_final_subcampaign_algos,
     regression_errors_max, regression_errors_sum, slot_dict) = experiment.perform(timesteps_stationary)

    # TODO: Context generation will be added

    ############################################
    # Plot results
    ############################################

    cumulative_stationary_reward = np.cumsum(stationary_rewards)
    # cumulative_context_generation_reward = np.cumsum(context_generation_rewards)
    cumulative_clairvoyant_reward = np.cumsum(clairvoyant_rewards)
    cumulative_disaggregated_clairvoyant_reward = np.cumsum(disaggregated_clairvoyant_rewards)

    # Cumulative rewards

    plt.plot(cumulative_stationary_reward)
    # plt.plot(cumulative_context_generation_reward)
    plt.plot(cumulative_clairvoyant_reward)
    plt.plot(cumulative_disaggregated_clairvoyant_reward)
    plt.legend(['GPTS - Stationary', 'Clairvoyant', 'Clairvoyant - Optimal context'],
               bbox_to_anchor=(1.05, 1), loc=2)
    plt.title("Cumulative reward")
    plt.savefig(plot_path + 'cumulative_reward.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Average regrets

    plt.plot([(cumulative_clairvoyant_reward[i] - cumulative_stationary_reward[i]) / (i + 1) for i in
              range(len(cumulative_stationary_reward))])
    # plt.plot(
    #     [(cumulative_disaggregated_clairvoyant_reward[i] - cumulative_context_generation_reward[i]) / (i + 1) for i in
    #      range(len(cumulative_context_generation_reward))])
    plt.legend(['GPTS - Stationary'], bbox_to_anchor=(1.05, 1), loc=2)
    plt.title("Average regret")
    plt.savefig(plot_path + 'average_regret.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Average percentage regrets

    plt.plot([((cumulative_clairvoyant_reward[i] - cumulative_stationary_reward[i]) / (i + 1)) / optimal_super_arm_value
              for i in range(len(cumulative_stationary_reward))])
    # plt.plot([((cumulative_disaggregated_clairvoyant_reward[i] - cumulative_context_generation_reward[i]) / (
    #         i + 1)) / optimal_disaggregated_super_arm_value
    #           for i in range(len(cumulative_context_generation_reward))])
    plt.legend(['GPTS - Stationary'], bbox_to_anchor=(1.05, 1), loc=2)
    plt.title("Average percentage regret")
    plt.savefig(plot_path + 'average_percentage_regret.png', bbox_inches='tight', dpi=300)
    plt.close()

    # TODO
    # GP estimations and number of pulls per arm - context_generation

    exit(0)
