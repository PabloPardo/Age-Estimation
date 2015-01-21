import matplotlib.cm as cm
import pylab
import numpy as np
from utils import cum_score_
from pandas.tools.plotting import scatter_matrix


def scatter_mat(x, y, fig_sz, diag, name, **kwargs):
    """
    Plot a Scatter Matrix of the data x.
    :param x:
    :param y:
    :param fig_sz:
    :param diag:
    :param name:
    :param kwargs:
    :return:
    """
    if not kwargs:
        kwargs = {'s': 30*np.ones(y.shape), 'c': y.values, 'cmap': cm.gist_stern}

    pylab.clf()
    scatter_matrix(x, alpha=1, figsize=(fig_sz, fig_sz), diagonal=diag, **kwargs)
    pylab.savefig(name)


def age_dist(y, n_bins, name):
    """
    Plot the age distribution of a given set y

    :return: The number of bins and its population.
    """
    pylab.figure()
    bins = range(n_bins+1)
    n, bins, patches = pylab.hist(y, bins, normed=1, histtype='bar', rwidth=0.8)
    pylab.show()
    pylab.savefig(name)

    return [n, bins, patches]


def cum_score(name, **kwargs):
    """
    Plots the Cumulative Score for the labels y and the prediction y_

    :return: Cumulative Score
    """
    if 'y' in kwargs and 'y_' in kwargs:
        y = kwargs['y']
        y_ = kwargs['y_']
    scores = kwargs['cum_score'] if 'cum_score' in kwargs else cum_score_(y, y_)

    pylab.figure()
    pylab.plot(range(11), scores)
    pylab.xlabel('Error Level (year)')
    pylab.ylabel('Cumulative Score (\%)')
    pylab.show()
    pylab.savefig(name)

    return scores