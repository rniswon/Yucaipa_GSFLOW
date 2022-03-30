import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


rain = np.random.rand(100000)
temp = np.random.rand(100000)
storage = np.power(rain,2.0) + temp
storage2 = np.power(rain,1.3) + temp

def plot_contour(xvalue, yvalue, zvalue, stat = 'mean', bins = 100, ax = None, c = 'k'):
    ret = stats.binned_statistic_2d(xvalue, yvalue, zvalue,stat, bins = bins)
    x = ret.x_edge[1:]
    y = ret.y_edge[1:]

    X, Y = np.meshgrid(x, y)

    #plt.contourf(X, Y, ret.statistic, cmap = 'jet')
    ax.contour(X, Y, ret.statistic, colors = c)

plt.figure()
plot_contour(rain, temp, storage-storage2, ax=plt.gca(), c = 'k')
plot_contour(rain, temp, storage2, ax=plt.gca(), c = 'b')

x = 1