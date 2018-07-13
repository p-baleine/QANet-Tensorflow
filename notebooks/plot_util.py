import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.signal import butter, filtfilt

colorlist = ["b", "c", "m", "r", "y", "g", "k"]

def plot_perplexity(*args, **kwargs):
    """学習時の経過ログをsmoothingしてプロットする
    参考:
      http://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    """
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    color_names = cycle(colorlist)
    cutoff = kwargs.get('cutoff', 1500)
    fs = kwargs.get('fs', 50000)
    steps_per_epoch = kwargs.get('steps_per_epoch', None)

    if kwargs.get('log', True):
        plt.yscale('log')

    max_x = 0

    for name, path in args:
        c = next(color_names)
        df = pd.read_csv(path)
        x, y = df['Step'], df['Value'].as_matrix()
        y_smooth = butter_lowpass_filtfilt(y, cutoff, fs)
        plt.plot(x, y, color=c, alpha=0.2, label=None)
        plt.plot(x, y_smooth, color=c, label=name)
        max_x = max(max_x, x.max())

    if steps_per_epoch is not None:
        for x in range(steps_per_epoch, max_x, steps_per_epoch):
            plt.plot((x, x + 1), (0, 1000), color='r')

    plt.legend()
