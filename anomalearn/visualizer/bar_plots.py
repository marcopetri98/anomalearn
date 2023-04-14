from numbers import Number
from typing import Tuple

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from ..input_validation import check_argument_types, is_matplotlib_color


def bar_plot(x_pos,
             bars_height,
             bars_width: float = 0.8,
             bars_colors: list = None,
             tick_labels: list = None,
             title: str = "",
             y_axis_label: str = "",
             x_axis_label: str = "",
             fig_size: Tuple = (8, 8),
             ax: Axes = None) -> None:
    """Makes a bar plot with the given parameters.
    
    Parameters
    ----------
    x_pos : array-like of shape (n_bars)
        The position of the bars.
    
    bars_height : array-like of shape (n_bars)
        The height of the bars to draw.
    
    bars_width : Number, default=0.8
        The width of the bars.
        
    bars_colors : color | list, default=None
        The color of the bars. The colors are the colors accepted by matplotlib.
        
    tick_labels : list[str], default=None
        The labels to write on the x-axis.
        
    title : str, default=""
        The title of the bar plot.
        
    y_axis_label : str, default=""
        The label on the y-axis for the plot.
        
    x_axis_label : str, default=""
        The label on the x-axis for the plot.
    
    fig_size : tuple, default=(8,8)
        The dimension of the figure to prompt.
        
    ax : Axes, default=None
        The axis on which to add the plot. If this is not None, the plot will be
        added to the axes, no new figure will be created and printed.

    Returns
    -------
    None
    
    Raises
    ------
    TypeError
        If any of the arguments has not the right type.
        
    ValueError
        If any of the arguments has wrong value.
    """
    x_pos = np.array(x_pos)
    bars_height = np.array(bars_height)

    check_argument_types([tick_labels, bars_width, title, y_axis_label, x_axis_label, fig_size, ax],
                         [[list, None], Number, str, str, str, tuple, [Axes, None]],
                         ["tick_labels", "bars_width", "title", "y_axis_label", "x_axis_label", "fig_size", "ax"])
    
    # check variable types
    if bars_colors is not None and not is_matplotlib_color(bars_colors):
        raise TypeError("bars_colors must be a valid matplotlib color")
    
    # check variable values
    if x_pos.ndim != 1 or bars_height.ndim != 1:
        raise ValueError("x_pos and bars_height must have one dimension")
    elif x_pos.shape != bars_height.shape:
        raise ValueError("x_pos and bars_height must have the same shape")
    elif isinstance(bars_colors, list) and len(bars_colors) != x_pos.shape[0]:
        raise ValueError("bars_colors must be one color or a list of the same "
                         "dimensions as x_pos")
    elif isinstance(tick_labels, list) and len(tick_labels) != x_pos.shape[0]:
        raise ValueError("tick_labels must be of the same dimensions as x_pos")
    
    # implementation
    if ax is None:
        _ = plt.figure(figsize=fig_size, tight_layout=True)
        
        plt.bar(x_pos,
                bars_height,
                bars_width,
                color=bars_colors,
                tick_label=tick_labels)
        plt.title(title)
        plt.ylabel(y_axis_label)
        plt.xlabel(x_axis_label)
        
        plt.show()
    else:
        ax.bar(x_pos,
               bars_height,
               bars_width,
               color=bars_colors,
               tick_label=tick_labels)
        ax.set_title(title)
        ax.set_ylabel(y_axis_label)
        ax.set_xlabel(x_axis_label)


def bar_class_distribution(classes,
                           labels: list[str] = None,
                           bars_width: float = 0.8,
                           bars_colors: list = None,
                           bars_width_step_fraction: float = 0.8,
                           title: str = "",
                           y_axis_label: str = "",
                           x_axis_label: str = "",
                           fig_size: Tuple = (8, 8),
                           ax: Axes = None) -> None:
    """Plots the class distribution using a bar plot.
    
    This function is a wrapper for the more generic bar_plot function which
    plots in general a bar plot given the plot information. This function
    automatically computes the
    
    Parameters
    ----------
    classes : array-like of shape (n_classes)
        An array having the dimension equal to the number of classes in which
        the elements are the number of examples for each class.
    
    labels : list[str], default=None
        The labels of the classes to write on the x-axis.
    
    bars_width : float, default=0.8
        The width of the bars.
        
    bars_colors : color | list, default=None
        The color of the bars. The colors are the colors accepted by matplotlib.
        
    bars_width_step_fraction : float, default=0.8
        It is the fraction `width/step` (we suggest values between 0 and 1), it
        is used to determine the x positions of the bar plot. It represents the
        amount of space that there must be between bars. If this variable is 0.8
        and bars_width is 0.8 the step will be 1. If this variable is 0.8 and
        bars_width is 1 the step will be 1.25 and so on.
        
    title : str, default=""
        The title of the bar plot.
        
    y_axis_label : str, default=""
        The label on the y-axis for the plot.
        
    x_axis_label : str, default=""
        The label on the x-axis for the plot.
    
    fig_size : tuple, default=(8,8)
        The dimension of the figure to prompt.
        
    ax : Axes, default=None
        The axis on which to add the plot. If this is not None, the plot will be
        added to the axes, no new figure will be created and printed.

    Returns
    -------
    None
    
    Raises
    ------
    TypeError
        If any of the arguments has not the right type.
        
    ValueError
        If any of the arguments has wrong value.
    """
    classes = np.array(classes)

    check_argument_types([labels, bars_width, bars_width_step_fraction, title, y_axis_label, x_axis_label, fig_size, ax],
                         [[list, None], Number, Number, str, str, str, tuple, [Axes, None]],
                         ["labels", "bars_width", "bars_width_step_fraction", "title", "y_axis_label", "x_axis_label", "fig_size", "ax"])
    
    # check types
    if bars_colors is not None and not is_matplotlib_color(bars_colors):
        raise TypeError("bars_colors must be a valid matplotlib color")
    
    # check values
    if classes.ndim != 1:
        raise ValueError("classes must have exactly one dimension")
    elif labels is not None and len(labels) != classes.shape[0]:
        raise ValueError("labels must have the same length of classes")
    elif isinstance(bars_colors, list) and len(bars_colors) != classes.shape[0]:
        raise ValueError("bars_colors must have the same length of classes")
    
    # implementation
    total = np.sum(classes)
    bars_height = classes / total
    
    step = bars_width / bars_width_step_fraction
    x_pos = np.arange(0, step * classes.shape[0], step)
    
    bar_plot(x_pos=x_pos,
             bars_height=bars_height,
             bars_width=bars_width,
             bars_colors=bars_colors,
             tick_labels=labels,
             title=title,
             y_axis_label=y_axis_label,
             x_axis_label=x_axis_label,
             fig_size=fig_size,
             ax=ax)
