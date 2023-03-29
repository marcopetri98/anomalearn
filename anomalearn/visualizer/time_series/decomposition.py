from numbers import Number
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt, gridspec

from .. import line_plot


def plot_time_series_decomposition(original,
                                   seasonal,
                                   trend,
                                   residual,
                                   x_ticks_loc=None,
                                   x_ticks_labels=None,
                                   x_ticks_rotation: float = 0,
                                   fig_size: Tuple = (16, 16)) -> None:
    """Plots in a single figure the original, seasonal, trend and residual.
    
    Parameters
    ----------
    original : array-like
        The original time series.
    
    seasonal : array-like
        The seasonal component of the time series.
    
    trend : array-like
        The trend component of the time series.
    
    residual : array-like
        The residual component of the time series.

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

    fig_size : Tuple
        The dimension of the figure.

    Returns
    -------
    None
    """
    if not isinstance(fig_size, tuple):
        raise TypeError("The fig_size parameter must be a tuple")
    
    if len(fig_size) != 2:
        raise ValueError("The fig_size parameter must have length 2")
    elif not isinstance(fig_size[0], Number) or not isinstance(fig_size[1], Number):
        raise ValueError("The fig_sice parameter must contain numbers")
    
    try:
        original = np.array(original)
        seasonal = np.array(seasonal)
        trend = np.array(trend)
        residual = np.array(residual)
    except Exception as e:
        raise ValueError("All the first four arguments must be array-like "
                         "objects. The original exception message was {}".format(e))

    fig = plt.figure(figsize=fig_size, tight_layout=True)
    gs = gridspec.GridSpec(4, 1)

    x_values = np.arange(original.shape[0])
    
    ax_obs = fig.add_subplot(gs[0, 0])
    line_plot(x_values,
              original,
              title="Observed",
              x_ticks_loc=x_ticks_loc,
              x_ticks_labels=x_ticks_labels,
              x_ticks_rotation=x_ticks_rotation,
              ax=ax_obs)
    
    ax_trn = fig.add_subplot(gs[1, 0])
    line_plot(x_values,
              trend,
              title="Trend",
              x_ticks_loc=x_ticks_loc,
              x_ticks_labels=x_ticks_labels,
              x_ticks_rotation=x_ticks_rotation,
              ax=ax_trn)
    
    ax_ses = fig.add_subplot(gs[2, 0])
    line_plot(x_values,
              seasonal,
              title="Seasonal",
              x_ticks_loc=x_ticks_loc,
              x_ticks_labels=x_ticks_labels,
              x_ticks_rotation=x_ticks_rotation,
              ax=ax_ses)
    
    ax_res = fig.add_subplot(gs[3, 0])
    line_plot(x_values,
              residual,
              title="Residual",
              x_ticks_loc=x_ticks_loc,
              x_ticks_labels=x_ticks_labels,
              x_ticks_rotation=x_ticks_rotation,
              ax=ax_res)
    
    plt.show()
