def prettify_super_arm(environment, super_arm, slot_dict):
    return str(
        [(environment.get_subcampaign(slot_dict[idx]['subcampaign']).name,
          slot_dict[idx]['slot'], arm) for
         (idx, arm) in super_arm])
