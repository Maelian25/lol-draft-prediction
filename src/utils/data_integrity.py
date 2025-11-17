import numpy as np

from src.utils.general_helper import save_file


def check_target_intergity(matches_states):
    """
    Provide a check on dataset to see if every target ban and pick are available
    (which they should be)
    """

    def is_target_available(row, target_col):
        target = row[target_col]
        if np.isnan(target):
            return True
        target = int(target)
        return row["champ_availability"][target] == 1

    # Columns check
    matches_states["pick_available"] = matches_states.apply(
        lambda row: is_target_available(row, "target_pick"), axis=1
    )
    matches_states["ban_available"] = matches_states.apply(
        lambda row: is_target_available(row, "target_ban"), axis=1
    )

    # Filter on the matches states file
    problem_rows = matches_states[
        ~matches_states["pick_available"] | ~matches_states["ban_available"]
    ]

    print(f"Unavailable targets: {len(problem_rows)}")

    # Save file
    save_file(problem_rows, location="", filename="targets_not_in_mask.csv")
