from numbers import Number
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..input_validation import check_argument_types, is_matplotlib_color


def box_plot(x,
             labels: list[str] = None,
             colors: list = None,
             boxes_pos=None,
             boxes_width: float = 0.5,
             title: str = "",
             y_axis_label: str = "",
             x_axis_label: str = "",
             fig_size: Tuple = (8, 8),
             ax: Axes = None) -> None:
    """Shows the box plot of the passed data.

    Parameters
    ----------
    x : array-like or list
        It is a 2D array-like or a list of 1D array-like. In the former case,
        the boxes are drawn computing mean and quartiles on each column. In the
        latter case, boxes are drawn computing mean and quartiles on each array
        of the list.

    labels : list of str, default=None
        The labels of the sequences on which the box plots are computed.

    colors : list[color] or color, default=None
        The colors of the lines to be drawn.

    boxes_pos : array-like, default=None
        It is the position at which the boxes must be positioned in the plot.

    boxes_width : float, default=0.5
        It is the width of the boxes on the plot.

    title : str, default=""
        The title of the plot.

    y_axis_label : str, default=""
        The label to print on the y-axis.

    x_axis_label : str, default=""
        The label to print on the x-axis.

    fig_size : tuple, default=(8,8)
        The dimension of the matplotlib figure to be drawn.

    ax : Axes, default=None
        The axis on which to add the plot. If this is not None, the plot will be
        added to the axes, no new figure will be created and printed.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        At least one of the arguments has been passed with the wrong type.

    ValueError
        At least one variable has unacceptable value or inconsistent value.
    """
    if isinstance(x, list):
        old_list = x.copy()
        x = []
        for arr in old_list:
            arr_np = np.array(arr)

            if arr_np.ndim != 1:
                raise ValueError("x must be a list of 1D arrays")

            x.append(arr_np)
    else:
        x = np.array(x)

        if x.ndim != 2:
            raise ValueError("x must be 2D array or a list of 1D arrays")

    box_num = len(x) if isinstance(x, list) else x.shape[1]
    if boxes_pos is None:
        boxes_pos = np.arange(box_num)
    else:
        boxes_pos = np.array(boxes_pos)

    check_argument_types([labels, boxes_width, title, y_axis_label, x_axis_label, fig_size, ax],
                         [[list, None], Number, str, str, str, Tuple, [Axes, None]],
                         ["labels", "boxes_width", "title", "y_axis_label", "x_axis_label", "fig_size", "ax"])

    if not isinstance(colors, list):
        colors = [colors] * (len(x) if isinstance(x, list) else x.shape[0])

    # check type
    if colors is not None and not is_matplotlib_color(colors):
        raise TypeError("colors must be a valid matplotlib color")

    # check values
    if boxes_pos.ndim != 1:
        raise ValueError("boxes_pos must be 1D array")
    elif box_num != boxes_pos.shape[0]:
        raise ValueError("boxes_pos must have a number of elements equal to "
                         "the number of sequences")
    elif colors is not None and box_num != len(colors):
        raise ValueError("colors must have the same length as the number of "
                         "boxes to draw (columns or length of x)")
    elif labels is not None and box_num != len(labels):
        raise ValueError("labels must have the same length as the number of "
                         "boxes to draw (columns or length of x)")

    # implementation
    if ax is None:
        fig = plt.figure(figsize=fig_size)

    # exploit identical interface between plt and ax to draw
    dwg = ax if ax is not None else plt
    if colors is None:
        dwg.boxplot(x,
                    positions=boxes_pos,
                    widths=boxes_width,
                    labels=labels)
    else:
        boxplot_ = dwg.boxplot(x,
                               positions=boxes_pos,
                               widths=boxes_width,
                               labels=labels,
                               patch_artist=True)
        for patch, color in zip(boxplot_["boxes"], colors):
            patch.set_facecolor(color)

    if ax is None:
        plt.title(title)
        plt.ylabel(y_axis_label)
        plt.xlabel(x_axis_label)

        plt.show()
    else:
        ax.set_title(title)
        ax.set_ylabel(y_axis_label)
        ax.set_xlabel(x_axis_label)
