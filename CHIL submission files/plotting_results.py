import warnings
import pandas as pd
import numpy as np
import random
import pickle

# plotting
import matplotlib.pyplot as plt
from matplotlib import ticker

plt.rc('font', size=16, family='serif')
plt.grid(zorder=-100)
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

np.random.seed(config.rnd_seed)

with open('data_N_1000.pickle', 'rb') as f:
    data_N_1000 = pickle.load(f)


def plot_one_bar(titles, lists, y_label):

    import matplotlib.cm as cm
    np.random.seed(0)
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.style.use('tableau-colorblind10')

    w = 0.12    # bar width
    y = [list(li) for li in lists]  # x-coordinates of your bars
    x = list(range(len(y)))

    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.grid(zorder=-100)
    plt.ylabel(y_label)

    ax.bar(x,
           height=[np.mean(y) for yi in y],
           # yerr=[np.std(yi) for yi in y],    # error bars
           capsize=10,  # error bar cap width in points
           width=w,    # bar width
           tick_label=[ti for ti in titles],

           color=(0, 0, 0, 0),  # face color transparent
           edgecolor=colors,
           # ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
           )

    for i in range(len(x)):

        c = list(range(len(y[i])))
        # distribute scatter randomly across whole width of bar
        scatter = ax.scatter(x[i] + np.random.random(len(y[i]))
                             * w - w / 2, y[i], c=c, cmap="coolwarm")

    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Treatments")

    ax.add_artist(legend1)
    plt.show()
