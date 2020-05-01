import os

import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(data):

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
    return n, bins, patches

def createDir(checkpoint_path):
    try:
        os.mkdir(checkpoint_path)
    except OSError:
        print("Creation of the directory %s failed" % checkpoint_path)
    else:
        print("Successfully created the directory %s " % checkpoint_path)