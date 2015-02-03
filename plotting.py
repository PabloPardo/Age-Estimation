import matplotlib.cm as cm
import pylab
import numpy as np
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
        kwargs = {'s': 30*np.ones(y.shape), 'c': y, 'cmap': cm.gist_stern}

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


def cum_score(name, *args, **kwargs):
    """
    Plots the Cumulative Score for the labels y and the prediction y_

    :return: Cumulative Score
    """

    pylab.figure()
    for arg in args:
        pylab.plot(range(26), arg,  linestyle='-', marker='o')
    pylab.xlabel('Error Level (year)')
    pylab.ylabel('Cumulative Score (%)')
    pylab.ylim((0, 100))
    if 'legend' in kwargs:
        pylab.legend(kwargs['legend'])
    pylab.grid()
    pylab.savefig(name, bbox_inches='tight')