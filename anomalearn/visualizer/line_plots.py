from numbers import Number
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..input_validation import check_argument_types, is_matplotlib_color


def __check_common_line_params(x,
                               y_lim: dict = None,
                               x_lim: dict = None,
                               x_ticks_loc=None,
                               x_ticks_labels=None,
                               x_ticks_rotation: float = 0,
                               formats: list[str] | str = None,
                               colors: list = None,
                               series_labels: list[str] | str = None,
                               title: str = "",
                               y_axis_label: str = "",
                               x_axis_label: str = "",
                               plot_legend: bool = True,
                               fig_size: Tuple = (8, 8),
                               ax: Axes = None) -> None:
    x_ticks_loc = np.array(x_ticks_loc) if x_ticks_loc is not None else None
    x_ticks_labels = np.array(x_ticks_labels) if x_ticks_labels is not None else None

    check_argument_types([y_lim, x_lim, x_ticks_rotation, formats, title, y_axis_label, x_axis_label, fig_size, ax],
                         [[dict, None], [dict, None], Number, [str, list, None], str, str, str, tuple, [Axes, None]],
                         ["y_lim", "x_lim", "x_ticks_rotation", "formats", "title", "y_axis_label", "x_axis_label", "fig_size", "ax"])
    
    # check that lims are numbers
    x_lim, y_lim = __set_lim(x_lim), __set_lim(y_lim)
    for key in ["low", "high"]:
        if x_lim is not None:
            check_argument_types([x_lim[key]],
                                 [[Number, None]],
                                 [f"x_lim[\"{key}\"]"])
        if y_lim is not None:
            check_argument_types([y_lim[key]],
                                 [[Number, None]],
                                 [f"y_lim[\"{key}\"]"])

    if colors is not None and not isinstance(colors, list):
        colors = [colors] * len(x) if isinstance(x, list) else colors

    # check type
    if colors is not None and not is_matplotlib_color(colors):
        raise TypeError("bars_colors must be a valid matplotlib color")
    elif not isinstance(plot_legend, bool):
        raise TypeError("plot_legend must be bool")

    # check values
    if isinstance(x, list):
        if formats is not None and len(x) < len(formats):
            raise ValueError("the number of formats must be at most equal to "
                             "the number of lines")
        elif colors is not None and len(x) < len(colors):
            raise ValueError("the number of colors must be at most equal to "
                             "the number of lines")
        elif series_labels is not None and len(x) < len(series_labels):
            raise ValueError("the number of labels must be equal to the number "
                             "of lines")
    elif series_labels is not None and not isinstance(series_labels, str):
        raise TypeError("series_labels must be a string if only one line has to"
                        " be plotted")
    elif formats is not None and not isinstance(formats, str):
        raise TypeError("if only one line is passed, format must be None or "
                        "a single format, not a list")
    elif isinstance(colors, list):
        raise ValueError("if only one line is passed, colors must be None or "
                         "a single color, not a list")
    elif x_ticks_loc is not None and x_ticks_labels is not None:
        if x_ticks_loc.shape != x_ticks_labels.shape:
            raise ValueError("if both x_ticks_loc and x_ticks_labels are passed"
                             ", they must have the same shape")
        elif x_ticks_loc.ndim != 1:
            raise ValueError("x_ticks_loc and x_ticks_labels must have at most "
                             "1 dimension")
    elif x_ticks_loc is not None and x_ticks_loc.ndim != 1:
        raise ValueError("x_ticks_loc must have at most 1 dimension")
    elif x_ticks_labels is not None and x_ticks_labels.ndim != 1:
        raise ValueError("x_ticks_labels must have at most 1 dimension")


def __set_lim(lim: dict = None) -> dict:
    """Checks correctness of lims and eventually corrects them.
    
    Parameters
    ----------
    lim : dict, default=None
        It represents the limits on an axis for the plot. The only keys used
        on the dictionary are "low" and "high" and they represent the limits
        on the axis.

    Returns
    -------
    correct_x_lim, correct_y_lim : dict | None, dict | None
        The correct dictionaries representing the lims.
    """
    new_lim = {"low": None, "high": None}
    
    if lim is not None and "low" in lim:
        new_lim["low"] = lim["low"]
    if lim is not None and "high" in lim:
        new_lim["high"] = lim["high"]
        
    return new_lim


def __line_plot(x,
                y,
                x_lim: dict = None,
                y_lim: dict = None,
                x_ticks_loc=None,
                x_ticks_labels=None,
                x_ticks_rotation: float = 0,
                formats: list[str] | str = None,
                colors: list = None,
                series_labels: list[str] | str = None,
                title: str = "",
                y_axis_label: str = "",
                x_axis_label: str = "",
                plot_legend: bool = True,
                fig_size: Tuple = (8, 8),
                ax: Axes = None,
                create_fig: bool = True,
                plot_fig: bool = True) -> None:
    """Creates a line plot from data.

    Parameters
    ----------
    x : array-like or list of array-like
        The data representing the independent variable to be used to create the
        line plot. It can be an array-like when only one line should be drawn,
        and a list of array-like in case multiple lines should be drawn. This
        arrays must contain numbers.

    y : array-like or list of array-like
        The data representing the dependent variable to be used to create the
        line plot. It can be an array-like when only one line should be drawn,
        and a list of array-like in case multiple lines should be drawn. All the
        array-like contained in this variable must have the same shape of the
        array-like contained in the `x` argument.
    
    x_lim : dict, default=None
        It represents the limits on the x-axis for the plot. The only keys used
        on the dictionary are "low" and "high" and they represent the limits
        on the axis.
        
    y_lim : dict, default=None
        It represents the limits on the y-axis for the plot. The only keys used
        on the dictionary are "low" and "high" and they represent the limits
        on the axis.

    x_ticks_loc : array-like, default=None
        The location at which printing the ticks labels. These will be also the
        labels in case the argument `x_ticks_labels` is None.

    x_ticks_labels : array-like, default=None
        The labels of the ticks on the x to be printed on the plot, they start
        at the first sample and end at the last sample if `x_ticks_loc` is None.
        Otherwise, they will be printed exactly at the position specified by the
        other argument.

    x_ticks_rotation : float, default=0.0
        The rotation of the ticks on the x-axis.

    formats : list[str] or str, default=None
        The formats for the lines to be drawn on the plot.

    colors : list[color] or color, default=None
        The colors of the lines to be drawn.

    series_labels: list[str] or str, default=None
        The labels of the lines to plot.

    title : str, default=""
        The title of the plot.

    y_axis_label : str, default=""
        The label to print on the y-axis.

    x_axis_label : str, default=""
        The label to print on the x-axis.

    plot_legend : bool, default=True
        States if the legend must be a plot on the line plot.

    fig_size : tuple, default=(8,8)
        The dimension of the matplotlib figure to be drawn.

    ax : Axes, default=None
        The axis on which to add the plot. If this is not None, the plot will be
        added to the axes, no new figure will be created and printed.

    create_fig : bool, default=True
        If `True` the function also creates the figure in case it is not called
        on an axis. Moreover, the title, and axis labels are set. Otherwise,
        the function does not create the figure and does not set title and axis
        labels. When an axis is passed and this is `False`, the title and axes
        labels are not set.

    plot_fig : bool, default=True
        If `True` the function also plots the figure in case it is not called on
        an axis. Otherwise, the function does not plot the figure.

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
    if isinstance(x, list) and not np.isscalar(x[0]):
        check_argument_types([y], [list], ["y"])

        if len(x) != len(y):
            raise ValueError("x and y must have the same dimension")

        for i, (var_ind, var_dep) in enumerate(zip(x, y)):
            var_dep = np.array(var_dep)
            var_ind = np.array(var_ind)
            x[i] = var_ind
            y[i] = var_dep

            if var_dep.shape != var_ind.shape:
                raise ValueError("the dependent and independent arrays must "
                                 "have the same shape. Found dependent shape "
                                 f"{var_dep.shape} and independent shape {var_ind.shape}")
            elif var_dep.ndim != 1:
                raise ValueError("when you pass a list of arrays, arrays must "
                                 f"have 1 dimension. Found dependent shape {var_dep.shape}")
    else:
        x = np.array(x)
        y = np.array(y)

        if x.shape != y.shape:
            raise ValueError(f"the dependent (shape {x.shape}) and independent "
                             f"(shape {y.shape}) arrays must have the same shape")
        elif x.ndim != 1:
            raise ValueError("arrays have more than one dimension, if you want "
                             "to plot multiple lines, pass a list of 1d arrays")

    __check_common_line_params(x=x,
                               x_lim=x_lim,
                               y_lim=y_lim,
                               x_ticks_loc=x_ticks_loc,
                               x_ticks_labels=x_ticks_labels,
                               x_ticks_rotation=x_ticks_rotation,
                               formats=formats,
                               colors=colors,
                               series_labels=series_labels,
                               title=title,
                               y_axis_label=y_axis_label,
                               x_axis_label=x_axis_label,
                               plot_legend=plot_legend,
                               fig_size=fig_size,
                               ax=ax)

    x_ticks_loc = np.array(x_ticks_loc) if x_ticks_loc is not None else None
    x_ticks_labels = np.array(x_ticks_labels) if x_ticks_labels is not None else None

    # implementation
    if ax is None and create_fig:
        fig = plt.figure(figsize=fig_size)

    # TODO: evaluate if this can be made top-level
    def add_line(ind, dep, line_color, line_format, axes, label):
        if label is not None:
            other_params = {"label": label}
        else:
            other_params = {}

        if axes is None:
            if line_format is not None:
                plt.plot(ind, dep, line_format, color=line_color, **other_params)
            else:
                plt.plot(ind, dep, color=line_color, **other_params)
        else:
            if line_format is not None:
                axes.plot(ind, dep, line_format, color=line_color, **other_params)
            else:
                axes.plot(ind, dep, color=line_color, **other_params)

    # TODO: evaluate if this can be made top-level
    def add_ticks(loc, label, rotation, axes):
        if axes is None:
            plt.xticks(loc, label, rotation=rotation)
        else:
            axes.set_xticks(loc, label, rotation=rotation)

    # plots all the lines
    if isinstance(x, list):
        for i, (var_ind, var_dep) in enumerate(zip(x, y)):
            line_fmt = None
            line_col = None
            if formats is not None and i < len(formats):
                line_fmt = formats[i]
            if colors is not None and i < len(colors):
                line_col = colors[i]

            label_ = None if series_labels is None else series_labels[i]

            add_line(var_ind, var_dep, line_col, line_fmt, ax, label_)
    else:
        add_line(x, y, colors, formats, ax, series_labels)

    # put ticks on the x if they are passed to the function
    if x_ticks_loc is not None or x_ticks_labels is not None:
        if x_ticks_loc is not None and x_ticks_labels is not None:
            # both are specified
            add_ticks(x_ticks_loc, x_ticks_labels, x_ticks_rotation, ax)
        elif x_ticks_loc is not None:
            # loc will also serve as label
            add_ticks(x_ticks_loc, x_ticks_loc, x_ticks_rotation, ax)
        else:
            # labels must go from the start to the end
            if isinstance(x, list):
                start = np.inf
                end = - np.inf
                for seq in x:
                    if np.min(seq) < start:
                        start = np.min(seq)
                    if np.max(seq) > end:
                        end = np.max(seq)
            else:
                start = np.min(x)
                end = np.max(x)
            x_ticks_loc = np.linspace(start, end, x_ticks_labels.shape[0])
            add_ticks(x_ticks_loc, x_ticks_labels, x_ticks_rotation, ax)

    x_lim, y_lim = __set_lim(x_lim), __set_lim(y_lim)

    if ax is None:
        plt.title(title)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.tight_layout()
        plt.ylim(y_lim["low"], y_lim["high"])
        plt.xlim(x_lim["low"], x_lim["high"])

        if series_labels is not None and plot_legend:
            plt.legend()

        if plot_fig:
            plt.show()
    else:
        ax.set_title(title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_ylim(y_lim["low"], y_lim["high"])
        ax.set_xlim(x_lim["low"], x_lim["high"])

        if series_labels is not None and plot_legend:
            ax.legend()


def line_plot(x,
              y,
              x_lim: dict = None,
              y_lim: dict = None,
              x_ticks_loc=None,
              x_ticks_labels=None,
              x_ticks_rotation: float = 0,
              formats: list[str] | str = None,
              colors: list = None,
              series_labels: list[str] | str = None,
              title: str = "",
              y_axis_label: str = "",
              x_axis_label: str = "",
              plot_legend: bool = True,
              fig_size: Tuple = (8, 8),
              ax: Axes = None) -> None:
    """Creates a line plot from data.

    Parameters
    ----------
    x : array-like or list of array-like
        The data representing the independent variable to be used to create the
        line plot. It can be an array-like when only one line should be drawn,
        and a list of array-like in case multiple lines should be drawn. This
        arrays must contain numbers.

    y : array-like or list of array-like
        The data representing the dependent variable to be used to create the
        line plot. It can be an array-like when only one line should be drawn,
        and a list of array-like in case multiple lines should be drawn. All the
        array-like contained in this variable must have the same shape of the
        array-like contained in the `x` argument.
    
    x_lim : dict, default=None
        It represents the limits on the x-axis for the plot. The only keys used
        on the dictionary are "low" and "high" and they represent the limits
        on the axis.
        
    y_lim : dict, default=None
        It represents the limits on the y-axis for the plot. The only keys used
        on the dictionary are "low" and "high" and they represent the limits
        on the axis.

    x_ticks_loc : array-like, default=None
        The location at which printing the ticks labels. These will be also the
        labels in case the argument `x_ticks_labels` is None.

    x_ticks_labels : array-like, default=None
        The labels of the ticks on the x to be printed on the plot, they start
        at the first sample and end at the last sample if `x_ticks_loc` is None.
        Otherwise, they will be printed exactly at the position specified by the
        other argument.
        
    x_ticks_rotation : float, default=0.0
        The rotation of the ticks on the x-axis.

    formats : list[str] or str, default=None
        The formats for the lines to be drawn on the plot.

    colors : list[color] or color, default=None
        The colors of the lines to be drawn.

    series_labels: list[str] or str, default=None
        The labels of the lines to plot.

    title : str, default=""
        The title of the plot.

    y_axis_label : str, default=""
        The label to print on the y-axis.

    x_axis_label : str, default=""
        The label to print on the x-axis.

    plot_legend : bool, default=True
        States if the legend must be a plot on the line plot.

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
    __line_plot(x=x,
                y=y,
                x_lim=x_lim,
                y_lim=y_lim,
                x_ticks_loc=x_ticks_loc,
                x_ticks_labels=x_ticks_labels,
                x_ticks_rotation=x_ticks_rotation,
                formats=formats,
                colors=colors,
                series_labels=series_labels,
                title=title,
                y_axis_label=y_axis_label,
                x_axis_label=x_axis_label,
                plot_legend=plot_legend,
                fig_size=fig_size,
                ax=ax,
                create_fig=True,
                plot_fig=True)


def confidence_line_plot(x,
                         y,
                         x_lim: dict = None,
                         y_lim: dict = None,
                         x_ticks_loc=None,
                         x_ticks_labels=None,
                         x_ticks_rotation: float = 0,
                         formats: list[str] | str = None,
                         colors: list = None,
                         series_labels: list[str] | str = None,
                         title: str = "",
                         y_axis_label: str = "",
                         x_axis_label: str = "",
                         plot_legend: bool = True,
                         conf_transparency: float = 0.5,
                         fig_size: Tuple = (8, 8),
                         ax: Axes = None) -> None:
    """Plots a line with its confidence region.

    Parameters
    ----------
    x : array-like or list of array-like
        The data representing the independent variable to be used to create the
        line plot. It can be an array-like when only one line should be drawn,
        and a list of array-like in case multiple lines should be drawn. This
        arrays must contain numbers.

    y : list array-like or list of lists of array-like
        A list of dimension 3 with the upper bounds at location 0, the lower
        bounds at location 1, and the estimate at location 2. Or a list of lists
        of dimension 3 composed as in the single case.
    
    x_lim : dict, default=None
        It represents the limits on the x-axis for the plot. The only keys used
        on the dictionary are "low" and "high" and they represent the limits
        on the axis.
        
    y_lim : dict, default=None
        It represents the limits on the y-axis for the plot. The only keys used
        on the dictionary are "low" and "high" and they represent the limits
        on the axis.

    x_ticks_loc : array-like, default=None
        The location at which printing the ticks labels. These will be also the
        labels in case the argument `x_ticks_labels` is None.

    x_ticks_labels : array-like, default=None
        The labels of the ticks on the x to be printed on the plot, they start
        at the first sample and end at the last sample if `x_ticks_loc` is None.
        Otherwise, they will be printed exactly at the position specified by the
        other argument.

    x_ticks_rotation : float, default=0.0
        The rotation of the ticks on the x-axis.

    formats : list[str] or str, default=None
        The formats for the lines to be drawn on the plot.

    colors : list[color] or color, default=None
        The colors of the lines to be drawn.

    series_labels: list[str] or str, default=None
        The labels of the lines to plot.

    title : str, default=""
        The title of the plot.

    y_axis_label : str, default=""
        The label to print on the y-axis.

    x_axis_label : str, default=""
        The label to print on the x-axis.

    plot_legend : bool, default=True
        States if the legend must be a plot on the line plot.

    conf_transparency : float, default=0.5
        The transparency of the confidence interval.

    fig_size : tuple, default=(8,8)
        The dimension of the matplotlib figure to be drawn.

    ax : Axes, default=None
        The axis on which to add the plot. If this is not None, the plot will be
        added to the axes, no new figure will be created and printed.

    Returns
    -------
    None
    """
    check_argument_types([y, conf_transparency],
                         [list, Number],
                         ["y", "conf_transparency"])

    if isinstance(x, list) and not np.isscalar(x[0]):
        if len(x) != len(y):
            raise ValueError("x and y must have the same dimension")

        for i, (var_ind, var_dep) in enumerate(zip(x, y)):
            if not isinstance(var_dep, list):
                raise ValueError("if you provide multiple lines, the y parameter"
                                 " must be a list of lists of length 3.")
            elif len(var_dep) != 3:
                raise ValueError("elements of y must be lists of length 3")
            
            var_ind = np.array(var_ind)
            lower_bound = np.array(var_dep[0])
            upper_bound = np.array(var_dep[1])
            estimate = np.array(var_dep[2])
            x[i] = var_ind
            y[i] = [lower_bound, upper_bound, estimate]

            if lower_bound.shape != upper_bound.shape or upper_bound.shape != estimate.shape:
                raise ValueError("elements of lists of y must have the same "
                                 "shape")
            elif not np.all(upper_bound > lower_bound):
                raise ValueError("the upper bound must always be greater than "
                                 "the lower bound")
            elif not np.all(upper_bound > estimate):
                raise ValueError("the estimate must be between upper and lower "
                                 "bounds (it is greater than upper bound for "
                                 "some points)")
            elif not np.all(estimate > lower_bound):
                raise ValueError("the estimate must be between upper and lower "
                                 "bounds (it is lower than lower bound for "
                                 "some points)")
            elif var_ind.shape != lower_bound.shape:
                raise ValueError("the dependent arrays must have the same shape"
                                 " as the elements of lists of y")
            elif var_ind.ndim != 1:
                raise ValueError("when you pass a list of arrays, arrays must "
                                 "have 1 dimension")
    else:
        x = np.array(x)
        y[0] = np.array(y[0])
        y[1] = np.array(y[1])
        y[2] = np.array(y[2])

        if x.shape != y[0].shape or y[0].shape != y[1].shape or y[1].shape != y[2].shape:
            raise ValueError("the dependent variable, lower bounds, upper "
                             "bounds and estimate must have the same shape")
        elif x.ndim != 1:
            raise ValueError("arrays have more than one dimension, if you want "
                             "to plot multiple lines, pass a list of 1d arrays")

    __check_common_line_params(x=x,
                               x_lim=x_lim,
                               y_lim=y_lim,
                               x_ticks_loc=x_ticks_loc,
                               x_ticks_labels=x_ticks_labels,
                               x_ticks_rotation=x_ticks_rotation,
                               formats=formats,
                               colors=colors,
                               series_labels=series_labels,
                               title=title,
                               y_axis_label=y_axis_label,
                               x_axis_label=x_axis_label,
                               plot_legend=plot_legend,
                               fig_size=fig_size,
                               ax=ax)

    if not 0 < conf_transparency < 1:
        raise ValueError("conf_transparency must be between 0 and 1")
    
    if ax is None:
        fig = plt.figure(figsize=fig_size)
    
    # TODO: see if it can be made top level
    def add_confidence_shadow(dep, lower, upper, conf_color, axes, alpha):
        if axes is None:
            plt.fill_between(dep, upper, lower, color=conf_color, alpha=alpha)
        else:
            axes.fill_between(dep, upper, lower, color=conf_color, alpha=alpha)
    
    # plot all lines and confidence bounds
    if isinstance(x, list):
        # add all the confidence shadows
        for i, (var_ind, var_dep) in enumerate(zip(x, y)):
            lower_bound = var_dep[0]
            upper_bound = var_dep[1]
            conf_color_ = None
            if colors is not None and i < len(colors):
                conf_color_ = colors[i]

            add_confidence_shadow(var_ind, lower_bound, upper_bound, conf_color_, ax, conf_transparency)

        # add the lines, ticks, legend and plot
        estimates = [e[2] for e in y]
        __line_plot(x=x,
                    y=estimates,
                    x_lim=x_lim,
                    y_lim=y_lim,
                    x_ticks_loc=x_ticks_loc,
                    x_ticks_labels=x_ticks_labels,
                    x_ticks_rotation=x_ticks_rotation,
                    formats=formats,
                    colors=colors,
                    series_labels=series_labels,
                    title=title,
                    y_axis_label=y_axis_label,
                    x_axis_label=x_axis_label,
                    plot_legend=plot_legend,
                    fig_size=fig_size,
                    ax=ax,
                    create_fig=True,
                    plot_fig=True)
    else:
        # add the confidence shadow
        add_confidence_shadow(x, y[0], y[1], colors, ax, conf_transparency)
        # add the line, ticks, legend and plot
        __line_plot(x=x,
                    y=y[2],
                    x_lim=x_lim,
                    y_lim=y_lim,
                    x_ticks_loc=x_ticks_loc,
                    x_ticks_labels=x_ticks_labels,
                    x_ticks_rotation=x_ticks_rotation,
                    formats=formats,
                    colors=colors,
                    series_labels=series_labels,
                    title=title,
                    y_axis_label=y_axis_label,
                    x_axis_label=x_axis_label,
                    plot_legend=plot_legend,
                    fig_size=fig_size,
                    ax=ax,
                    create_fig=False,
                    plot_fig=True)
