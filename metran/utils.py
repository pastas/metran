def show_versions():
    """Method to print the version of dependencies."""
    from sys import version as os_version

    from matplotlib import __version__ as mpl_version
    from numpy import __version__ as np_version
    from pandas import __version__ as pd_version
    from pastas import __version__ as ps_version
    from scipy import __version__ as sc_version

    msg = (
        f"Python version: {os_version}\n"
        f"numpy version: {np_version}\n"
        f"scipy version: {sc_version}\n"
        f"pandas version: {pd_version}\n"
        f"matplotlib version: {mpl_version}\n"
        f"pastas version: {ps_version}"
    )

    # numba
    try:
        from numba import __version__ as nb_version
        msg = msg + f"\nnumba version: {nb_version}"
    except ModuleNotFoundError:
        msg = msg + "\nnumba version: not installed"

    # lmfit
    try:
        from lmfit import __version__ as lm_version
        msg = msg + f"\nlmfit version: {lm_version}"
    except ModuleNotFoundError:
        msg = msg + "\nlmfit version: not installed"

    return print(msg)
