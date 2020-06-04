from algorithms.knapsack import Knapsack
from experiment.subcampaign_algo import Slot_algo
from utilities.rounding import round_to_nearest_feasible_superarm
from utilities.print_functions import prettify_super_arm


class Experiment:
    def __init__(self, environment, budget_discretization_steps, daily_budget, GPTS_prior):
        self.environment = environment
        self.budget_discretization_steps = budget_discretization_steps
        self.daily_budget = daily_budget
        self.GPTS_prior = GPTS_prior

    def perform(self, timesteps, context_generation_rate=-1):
        rewards = []
        environment = self.environment.copy()
        num_slots = environment.get_number_of_slots()
        slot_algos = []
        slot_dict = []
        for subcampaign in self.environment.subcampaigns:
            num_interested_users = subcampaign.sample(save_sample=False)
            for slot in subcampaign.slots:
                slot_algo = Slot_algo(subcampaign.id, slot.id, self.budget_discretization_steps.copy(), self.GPTS_prior)
                samples = [[k * self.daily_budget / 50 for k in range(51)],
                           [sum(slot.sample_for_all_classes(
                               k * self.daily_budget / 50,
                               interested_users=num_interested_users, save_sample=False)) for k in range(51)]]
                slot_algo.learn_gp_kernel_hyperparameters(samples)
                slot_algos.append(slot_algo)
                slot_dict.appen({'subcampaign': subcampaign.id,
                                 'slot': slot.id
                                 })

        # Run only when without context generation
        if context_generation_rate < 0:
            regression_errors_max = [[] for _ in range(num_slots)]
            regression_errors_sum = [[] for _ in range(num_slots)]

        for t in range(timesteps):

            # Sample from the subcampaign_algos to get clicks estimations
            estimations = []
            for slot_algo in slot_algos:
                estimate = [slot_algo.sample_from_gp(arm) for arm in self.budget_discretization_steps]
                if sum(estimate) == 0:
                    estimate = [i * 1e-3 for i in range(len(self.budget_discretization_steps))]
                estimate[0] = 0
                estimations.append(estimate)

            # Run knapsack optimization
            super_arm = Knapsack(self.daily_budget, estimations).optimize()

            # Fix for first day
            if t == 0:
                super_arm = [(i, self.daily_budget / num_slots) for i in range(num_slots)]
                super_arm = round_to_nearest_feasible_superarm(super_arm, self.budget_discretization_steps)

            # Collect rewards and update slot_algos
            total_reward = 0
            for (slot_idx, budget_assigned) in super_arm:
                subcampaign_id = slot_dict[slot_idx]['subcampaign']
                slot_id = slot_dict[slot_idx]['slot']
                num_interested_users = environment.get_subcampaign(subcampaign_id).sample(save_sample=False)

                reward = environment.get_subcampaign(subcampaign_id).get_slot(slot_id).sample_for_all_classes(
                    budget_assigned, interested_users=num_interested_users, save_sample=False)
                total_reward += sum(reward)

                # Fit multiple point to the GPs (one per each class of user inside this subcampaing)
                slot_algos[slot_idx].update(budget_assigned, reward)

            print("-------------------------")
            print("t = " + str(t + 1) + ", superarm = "
                  + prettify_super_arm(environment,
                                       super_arm,
                                       slot_dict) + ", reward = " + str(total_reward))

            rewards.append(total_reward)

            # Run only when without context generation
            if context_generation_rate < 0:
                for i in range(num_slots):
                    subcampaign_id = slot_dict[i]['subcampaign']
                    slot_id = slot_dict[i]['slot']
                    regression_rewards = [(x,
                                           environment.subcampaigns[subcampaign_id].get_slot(slot_id).get_real(x))
                                          for x in
                                          self.budget_discretization_steps]
                    regression_errors_max[i].append(
                        slot_algos[i].get_regression_error(points_to_evaluate=regression_rewards))
                    regression_errors_sum[i].append(
                        slot_algos[i].get_regression_error(use_sum=True, points_to_evaluate=regression_rewards))

            print("-------------------------")

            if context_generation_rate > 0:
                return (rewards, environment, slot_algos, slot_dict)
            else:
                return (rewards, environment, slot_algos, regression_errors_max, regression_errors_sum, slot_dict)
