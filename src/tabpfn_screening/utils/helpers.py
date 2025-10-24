"""Helper utility functions."""

import re


def slug(s):
    """
    Convert a string to a filesystem-safe slug.
    
    Parameters
    ----------
    s : str
        Input string
        
    Returns
    -------
    str
        Slugified string safe for filenames
    """
    return re.sub(r"[^-a-zA-Z0-9_.]+", "_", str(s)).strip("_")

