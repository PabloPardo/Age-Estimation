import numpy as np
import os.path
from sklearn.decomposition.pca import PCA


def group_y(y, groups):
    """
    Return the labels splitted between a given label groups
    :param y: Labels
    :param groups: Groups to split the labels
    :return: Splitted labels
    """
    y_ = []
    for i in y:
        num_g = 0
        for g in groups:
            if i < g:
                y_.append(int(num_g))
                break
            num_g += 1
    return np.array(y_)


def mae_score(est, x, y):
    """
    Calculate the Mean Absolute Error (MAE)

    :param est: Estimator
    :param x: Data
    :param y: Targets
    :return: MAE
    """
    y_ = est.predict(x)
    return mae_score_(y, y_)


def mae_score_(y, y_):
    return np.mean(np.absolute(np.subtract(y, y_)))


def cum_score_(y, y_):
    err = np.absolute(np.subtract(y, y_))
    return [np.mean([e for e in err if e >= i])*100 for i in range(0, 11)]


def acc_score(est, x, y):
    """
    Calculate the Accuracy

    :param est: Estimator
    :param x: Data
    :param y: Targets
    :return: Accuracy
    """
    y_ = est.predict(x)
    return sum([1 for i in range(len(y_)) if y_[i] == y[i]])/float(len(y))


def do_pca(x, var, name):
    """
    Normalizes the data x, computes PCA from data x keeping
    the variance var, normalizes the projected data again.
    It save projected data into a csv file with the specified
    name.

    :param x: Data
    :param var: Variance to keep (from 0 - 1)
    :param name: Name of the csv file to save
    :return: Projected data
    """
    x = x.transpose()
    x = (x - x.mean(axis=0))/x.std(axis=0)

    pca = PCA(n_components=var)
    proj = pca.fit_transform(x)
    proj = (proj - proj.mean(axis=0))/proj.std(axis=0)
    np.savetxt(name, proj, delimiter=',')

    return proj


def outliers_filter(v, margin):
    """
    Filters a numeric vector to remove the outliers.
    It calculates the median of the array and keeps
    the elements within a range up to 'margin' distance
    from the median.

    :param v: Array to filter
    :param margin: Margin from where discard the elements.
    :return: Filtered array
    """
    median = np.median(v)
    return [item for item in v if abs(item - median) <= margin]


def load_data(dataset_name):
    """
    Load Data
    """
    if dataset_name == 'FGNET':
        y = np.genfromtxt('data/FGNET_Y.csv', delimiter=',')

        aux = np.genfromtxt('../../Databases/Aging DB/FGNET/ages.csv', delimiter=',', usecols=(0, 1, 3))
        gender = aux[:, 2]
        ind = aux[:, 1]

        if not os.path.exists('data/FGNET_pca_0.95_X.csv') and not os.path.exists('data/FGNET_pca_0.95_shapes.csv'):
            x = np.genfromtxt('data/FGNET_X.csv', delimiter=',')
            shapes = np.genfromtxt('data/FGNET_shapes.csv', delimiter=',')

            # Normalize and dimensionality reduction
            proj_x = do_pca(x, 0.95, 'data/FGNET_pca_0.95_X.csv')
            proj_sh = do_pca(shapes, 0.95, 'data/FGNET_pca_0.95_shapes.csv')
        else:
            proj_x = np.genfromtxt('data/FGNET_pca_0.95_X.csv', delimiter=',')
            proj_sh = np.genfromtxt('data/FGNET_pca_0.95_shapes.csv', delimiter=',')
    elif dataset_name == 'HuPBA':
        pass
    else:
        raise NameError('Dataset %s not found.' % dataset_name)

    return proj_x, y, proj_sh, gender, ind