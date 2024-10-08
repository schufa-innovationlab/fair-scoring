"""
Internal utilities to generate the plots.
"""

import numpy as np

import matplotlib.colors as mc
import colorsys


def lighten_color(color, amount=0.5, return_as_hex=False):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Notes
    -----
    Taken from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Examples
    --------
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    res = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    if return_as_hex:
        return mc.to_hex(res)
    else:
        return res