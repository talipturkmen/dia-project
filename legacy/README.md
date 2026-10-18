# Original 2020 code

This folder holds the team's original 2020 implementation, unchanged, for reference. It is not used by the `dia` package.

Known problems that the 2026 rewrite fixed:

- `experiment/experiment.py`: the `return` sits inside the time loop, so every experiment stops after one day.
- `algorithms/hungarian_algorithm.py`: `step2` reads a global `a` that only exists under `__main__`, so the function fails when imported.
- `environment/classes.py`: the slot-prominence loop multiplies by values it has already overwritten, and modifies the shared configuration in place.
- `learner/GPTS.py`: `get_average_regression_error` calls `len()` on a `zip` object.
- Points 3, 4, 5 and 7 of the assignment were not implemented. Point 6 split the budget across slots rather than across subcampaigns, and did not choose bids.
