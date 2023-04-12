from matplotlib.colors import is_color_like


def is_matplotlib_color(color) -> bool:
    """Checks whether it is a color accepted by matplotlib.

    Parameters
    ----------
    color : color or list[color]
        A matplotlib color or a list of matplotlib colors.

    Returns
    -------
    is_matplotlib_color : bool
        True whether it is a matplotlib color or a valid list of matplotlib
        colors.
    """
    if isinstance(color, list):
        for col in color:
            if not is_color_like(col):
                return False
        return True
    else:
        return is_color_like(color)
