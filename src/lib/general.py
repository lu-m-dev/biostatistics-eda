"""This module contains general utility functions."""


def get_stage_list(n: int = 2) -> list[str]:
    """
    Get the list of stages for a given version.

    Parameters
    ----------
    n : int, optional
        number of bits, by default 2

    Returns
    -------
    list[str]
        stage list
    """
    stages: dict[int, list[str]] = {
        3: [
            "CSF-/PET-/pTau-",
            "CSF+/PET-/pTau-",
            "CSF-/PET+/pTau-",
            "CSF+/PET+/pTau-",
            "CSF-/PET+/pTau+",
            "CSF+/PET+/pTau+",
            "CSF-/PET-/pTau+",
        ],
        2: [
            "CSF-/PET-",
            "CSF+/PET-",
            "CSF-/PET+",
            "CSF+/PET+",
        ],
    }
    return stages[n]
