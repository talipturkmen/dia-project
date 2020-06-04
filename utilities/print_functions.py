def prettify_super_arm(environment, super_arm, slot_dict):
    return str(
        [(environment.get_subcampaign(slot_dict[idx]['subcampaign']).name,
          environment.get_subcampaign(slot_dict[idx]['subcampaign']).get_slot(slot_dict[idx]['slot']).id, arm) for
         (idx, arm) in super_arm])
