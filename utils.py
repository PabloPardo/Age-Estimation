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
    if type(y) == list:
        return sum([1 for i in range(len(y_)) if y_[i] == y[i]])/float(len(y_))
    else:
        return sum([1 for i in range(len(y_)) if y_[i] == y])


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


def load_data(dataset_name, type_label='apparent'):
    """
    Load Data
    """
    if dataset_name == 'FGNET':
        path = 'data/FG-NET/'
        path_csv = '../../Databases/Aging DB/FGNET/ages.csv'
        has_gender = True
        has_ind = True
    elif dataset_name == 'HuPBA':
        path = 'data/HuPBA/'
        path_csv = '../../Databases/Aging DB/AGE HuPBA/HuPBA_AGE_data_extended.csv'
        has_gender = False
        has_ind = False
    else:
        raise NameError('Dataset %s not found.' % dataset_name)

    y = np.genfromtxt(path + dataset_name + '_Y.csv', delimiter=',')

    if type_label == 'real':
        y = y[:, 0]
    elif type_label == 'apparent':
        y = y[:, 1]
    else:
        raise NameError('The type of label must be "real" or "apparent"')

    aux = np.genfromtxt(path_csv, delimiter=',', usecols=(0, 1, 3))
    if has_gender:
        gender = aux[:, 2]
    else:
        gender = []
    if has_ind:
        ind = aux[:, 1]
    else:
        ind = []

    if not os.path.exists(path + dataset_name + '_pca_0.95_X.csv') and not os.path.exists(path + dataset_name + '_pca_0.95_shapes.csv'):
        x = np.genfromtxt(path + dataset_name + '_X.csv', delimiter=',')
        shapes = np.genfromtxt(path + dataset_name + '_shapes.csv', delimiter=',')

        # Normalize and dimensionality reduction
        proj_x = do_pca(x, 0.95, path + dataset_name + '_pca_0.95_X.csv')
        proj_sh = do_pca(shapes, 0.95, path + dataset_name + '_pca_0.95_shapes.csv')
    else:
        proj_x = np.genfromtxt(path + dataset_name + '_pca_0.95_X.csv', delimiter=',')
        proj_sh = np.genfromtxt(path + dataset_name + '_pca_0.95_shapes.csv', delimiter=',')

    return proj_x, y, proj_sh, gender, ind