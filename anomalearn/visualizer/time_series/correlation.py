from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_correlation_functions(func_values: dict,
                               func_conf: dict,
                               fig_size: Tuple = (12, 12)) -> None:
    """Plots the ACF, PACF or both functions.
    
    Both dictionaries are meant to have string keys ("ACF" or "PACF") and the
    values for the keys must be array-like objects.
    
    Parameters
    ----------
    func_values : dict
        Dictionary containing the ACF and/or PACF values. Keys can be either
        "ACF" or "PACF".
    
    func_conf : dict
        Dictionary conaining the ACF and/or PACF confidence values. Keys can be
        either "ACF" or "PACF".
        
    fig_size : Tuple, default=(6,6)
        It is the default size for the matplotlib figure.

    Returns
    -------
    None
    """
    if "ACF" not in func_values.keys() and "PACF" not in func_values.keys():
        raise ValueError("Either ACF or PACF must be passed")
    
    if len(set(func_values.keys()).difference(func_conf.keys())) != 0:
        raise ValueError("Keys of the dictionaries must exactly match")
    
    for key in func_values.keys():
        func_values[key] = np.array(func_values[key])
        
    for key in func_conf.keys():
        func_values[key] = np.array(func_values[key])
    
    # plot correlation values
    if len(func_values.keys()) == 1:
        if "ACF" in func_values.keys():
            values: np.ndarray = func_values["ACF"]
            conf: np.ndarray = func_conf["ACF"]
        else:
            values: np.ndarray = func_values["PACF"]
            conf: np.ndarray = func_conf["PACF"]
            
        name = list(func_values.keys())[0]
        
        lags = values.shape[0] - 1
        _ = plt.figure(figsize=fig_size)
        plt.title(f"{name} function")
        plt.ylim(-1, 1)
        plt.plot([0, lags], [0, 0], linewidth=0.5)
        plt.vlines(range(lags + 1), [0] * (lags + 1), values)
        plt.scatter(range(lags + 1), values)
        plt.fill_between(np.linspace(1, lags, lags),
                         conf[1:, 0] - values[1:],
                         conf[1:, 1] - values[1:],
                         alpha=0.25,
                         linewidth=0.5)
        plt.show()
    else:
        acf_values: np.ndarray = func_values["ACF"]
        acf_conf: np.ndarray = func_conf["ACF"]
        pacf_values: np.ndarray = func_values["PACF"]
        pacf_conf: np.ndarray = func_conf["PACF"]

        lags = acf_values.shape[0] - 1
        _, axs = plt.subplots(2, 1, figsize=(12, 12))
        axs[0].set_title("ACF function")
        axs[0].set_ylim(-1, 1)
        axs[0].plot([0, lags], [0, 0], linewidth=0.5)
        axs[0].vlines(range(lags + 1), [0] * (lags + 1), acf_values)
        axs[0].scatter(range(lags + 1), acf_values)
        axs[0].fill_between(np.linspace(1, lags, lags),
                            acf_conf[1:, 0] - acf_values[1:],
                            acf_conf[1:, 1] - acf_values[1:],
                            alpha=0.25,
                            linewidth=0.5)

        axs[1].set_title("PACF function")
        axs[1].set_ylim(-1, 1)
        axs[1].plot([0, lags], [0, 0], linewidth=0.5)
        axs[1].vlines(range(lags + 1), [0] * (lags + 1), pacf_values)
        axs[1].scatter(range(lags + 1), pacf_values)
        axs[1].fill_between(np.linspace(1, lags, lags),
                            pacf_conf[1:, 0] - pacf_values[1:],
                            pacf_conf[1:, 1] - pacf_values[1:],
                            alpha=0.25,
                            linewidth=0.5)
        plt.show()
