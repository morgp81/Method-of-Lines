import numpy as np
def godunov(uL, uR):
    """
    Computes the Godunov flux for the Burgers' equation given left and right states.

    Parameters:
    uL : float
        The value of the solution to the left of the interface.
    uR : float
        The value of the solution to the right of the interface.

    Returns:
    flux : float
        The Godunov flux at the interface.
    """
    if (uL <= 0) and (uR >= 0):
        flux = 0.0
    elif (uL >= 0) and (uR >= 0):
        flux = uL**2 / 2
    elif (uL + uR) / 2 >= 0:
        flux = uL**2 / 2
    elif (uL + uR) / 2 < 0:
        flux = uR**2 / 2
    elif (uL <= 0) and (uR <= 0):
        flux = uR**2 / 2
    return flux