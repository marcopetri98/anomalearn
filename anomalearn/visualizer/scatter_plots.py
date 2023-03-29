from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle

from ..input_validation import check_argument_types, is_matplotlib_color


def scatter_plot(x,
                 y,
                 size: float | list = None,
                 colors: list = None,
                 markers: str | MarkerStyle | list[str | MarkerStyle] = "o",
                 labels: list[str] | str = None,
                 title: str = "",
                 y_axis_label: str = "",
                 x_axis_label: str = "",
                 fig_size: Tuple = (8, 8),
                 ax: Axes = None) -> None:
    """Makes a scatter plot using matplotlib.
    
    Parameters
    ----------
    x : array-like of shape (n,) or list of array-like
        The positions on the x-axis of the points or a list of the groups of
        points to plot.
    
    y : array-like of shape (n,) or list of array-like
        The positions on the y-axis of the points or a list of the groups of
        points to plot.
    
    size : float or array-like or list of floats or array-like, default=None
        It is the size of the points specified as `points**2`. If it is a float
        it represents the size of all the points, otherwise, it has the same
        shape as x and represents the size of each point. If there are groups
        of points, it must be a list of float or list of array-like.
    
    colors : color or list of colors or array-like or list of the previous, default=None
        The colors of the points to draw on the scatter plot. If the function
        receives as input points and not list of points, and if the color is an
        array-like of colors or a list of colors, it must have the same
        dimensions as `x` and `y`. The options for this variable are:
        - color matplotlib color for all the points
        - list of colors: list of colors having the same length of x
        - 2D array: array of shape (n, 3 or 4) showing the rgb or rgba of points
        Differently, if in input we have groups of points (list of array-like),
        colors must be a color (equal for all groups), list of colors, list of
        lists of colors or array-like. In such a case, each element of
        the list must respect the same constraints as in the single group in
        input (x and y are arrays).
    
    markers : str or MarkerStyle or list[str or MarkerStyle], default="o"
        It is the marker for the points to draw on the scatter plot. In case in
        input we have groups of points, it is the list of markers for each
        group.
        
    labels : list of str or str, default=None
        The labels of the scatter points. If this is None no legend will be
        printed, if this has values a legend will be printed.

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
        If any of the arguments has a wrong type.
    
    ValueError
        If any of the arguments has an invalid value.
    """
    if isinstance(x, list) and not np.isscalar(x[0]):
        check_argument_types([y], [list], ["y"])
        
        if len(x) != len(y):
            raise ValueError("x and y must have the same number of groups")
    
        for i, couple in enumerate(zip(x, y)):
            x_val, y_val = np.array(couple[0]), np.array(couple[1])
            x[i] = x_val
            y[i] = y_val
        
            if x_val.ndim != 1 or y_val.ndim != 1 or x_val.shape != y_val.shape:
                raise ValueError("arrays of x and y lists must have the same "
                                 "shape, and they must be 1D arrays")
    else:
        x = np.array(x)
        y = np.array(y)
    
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError("x and y must be 1D array with the same shape")
    
    check_argument_types([labels, markers, title, y_axis_label, x_axis_label, fig_size, ax],
                         [[list, str, None], [list, str, MarkerStyle], str, str, str, Tuple, [Axes, None]],
                         ["labels", "marker", "title", "y_axis_label", "x_axis_label", "fig_size", "ax"])
    
    if isinstance(x, list):
        # check labels
        if labels is not None and not isinstance(labels, list):
            raise TypeError("labels must be None or a list if x is a list")
        elif labels is not None and len(labels) != len(x):
            raise ValueError("labels must have the same length as x")
        
        # check markers type
        if isinstance(markers, str):
            markers = [markers] * len(x)
        elif len(markers) != len(x):
            raise ValueError("markers has not the same length of x")
            
        for marker in markers:
            check_argument_types([marker], [[str, MarkerStyle]], ["element of markers"])
        
        # check that each element of size is coherent with ith element of x
        if size is not None and not isinstance(size, list):
            raise ValueError("if x is a list, size can be a list or None")
        elif isinstance(size, list):
            if len(size) != len(x):
                raise ValueError("size must have the same length as x and y")
            
            for i, size_val in enumerate(size):
                if size_val is not None and not isinstance(size_val, float):
                    size_val = np.array(size_val)
                    size[i] = size_val
                    
                    if size_val.shape != x[i].shape:
                        raise ValueError("size elements must be None, float or "
                                         "arrays with shape consistent with "
                                         "that of arrays in x and y")
                else:
                    size[i] = size_val
        else:
            size = [None] * len(x)
            
        # check that each element of colors is coherent with ith element of x
        if colors is None:
            colors = [None] * len(x)
        elif is_matplotlib_color(colors):
            if not isinstance(colors, list):
                colors = [colors] * len(x)
        elif isinstance(colors, list):
            # check that each element is either a list of colors or a 2D array
            for i, col in enumerate(colors):
                if isinstance(col, list):
                    if not is_matplotlib_color(col):
                        raise ValueError("an element of colors is a list, but "
                                         "not a valid list of matplotlib colors")
                    elif len(col) != x[i].shape[0]:
                        raise ValueError("if a list of colors is passed, it "
                                         "must have the same number of elements"
                                         " of the elements in the group it "
                                         "represents")
                else:
                    col = np.array(col)
                    
                    if col.ndim != 2:
                        raise ValueError("arrays of colors must be 2D")
                    elif col.shape[0] != x[i].shape[0]:
                        raise ValueError("arrays of colors must have the same "
                                         "number of elements as the group they "
                                         "represent")
                    elif col.shape[1] != 3 or col.shape[1] != 4:
                        raise ValueError("arrays of colors must contain RGB or "
                                         "RGBA colors -> shape (n, 3 or 4)")
        else:
            raise ValueError("colors has wrong type and or value. Since x is "
                             "a list, colors must be: a color, a list of colors"
                             ", a list of lists of colors or arrays")
    else:
        # check that labels is a string
        if labels is not None and not isinstance(labels, str):
            raise TypeError("labels must be a string if only one group has been"
                            " provided")
        
        # check markers type
        check_argument_types([markers], [[str, MarkerStyle]], ["markers"])
        
        # check that size has the correct value and type
        if size is not None and not isinstance(size, float):
            size = np.array(size)
            
            if size.shape != x.shape:
                raise ValueError("size must have the same shape as x if it is "
                                 "an array")
            
        # check that colors has the correct value
        if colors is not None and not is_matplotlib_color(colors):
            colors = np.array(colors)
            
            if colors.ndim != 2:
                raise ValueError("colors must have 2 dimensions if it is an "
                                 "array")
            elif colors.shape[0] != x.shape[0]:
                raise ValueError("colors and x must have the same shape on the "
                                 "first axis (n. colors = n. points)")
            elif colors.shape[1] != 3 and colors.shape[1] != 4:
                raise ValueError("colors must be either RGB or RGBA -> shape "
                                 "(n, 3/4)")
        
    # implementation
    if ax is None:
        fig = plt.figure(figsize=fig_size)
        
    dwg = ax if ax is not None else plt
    if not isinstance(x, list):
        if labels is not None:
            dwg.scatter(x,
                        y,
                        s=size,
                        c=colors,
                        marker=markers,
                        label=labels)
        else:
            dwg.scatter(x,
                        y,
                        s=size,
                        c=colors,
                        marker=markers)
    else:
        if labels is not None:
            for x_val, y_val, s, c, m, l in zip(x, y, size, colors, markers, labels):
                dwg.scatter(x_val,
                            y_val,
                            s=s,
                            c=c,
                            marker=m,
                            label=l)
        else:
            for x_val, y_val, s, c, m in zip(x, y, size, colors, markers):
                dwg.scatter(x_val,
                            y_val,
                            s=s,
                            c=c,
                            marker=m)
        
    if ax is None:
        plt.title(title)
        plt.ylabel(y_axis_label)
        plt.xlabel(x_axis_label)
        if labels is not None:
            plt.legend()
        
        plt.show()
    else:
        ax.set_title(title)
        ax.set_ylabel(y_axis_label)
        ax.set_xlabel(x_axis_label)
        if labels:
            ax.legend()
