import time
from utils import *
from plotting import *
from joblib import Parallel, delayed
from sklearn import grid_search, svm


def validate(x, y, estimator, ind, i, evel_func):
    """
    Train the estimator and test with the individual i, returns the socre
    given by the evaluation function.

    :param x: Input data
    :param y: Target
    :param estimator: Estimator
    :param ind: List of all the input data labelled by individual
    :param i: Individual for test
    :param evel_func: Evaluation Function
    :return: Score
    """
    if len(ind) > 0:
        train_idx = [j for j in range(len(x)) if not ind[j] == i]
        test_idx = [j for j in range(len(x)) if ind[j] == i]
    else:
        train_idx = [j for j in range(len(x)) if not j == i]
        test_idx = i

    # Train Estimator
    estimator.fit(x[train_idx], y[train_idx])

    # Test Estimator
    return evel_func(estimator, x[test_idx], y[test_idx])


def group_class_lopo(x, y, ind, **kwargs):
    print 'Age group Classification ...'
    # Get group labels
    age_bins = [15, 40, 100]
    y_gr = group_y(y, age_bins)

    # Parameter Search
    c_param = np.array(range(180, 200))/1000.0 if 'c' not in kwargs else kwargs['c']

    best_acc = np.nan
    best_std = 0
    best_param = []
    best_model = []
    for c in c_param:
        time_el = time.time()
        # Create estimator instance
        svc = svm.SVC(kernel='linear', C=c)

        # Leave One Person Out (LOPO)
        if len(ind) > 0:
            acc = Parallel(n_jobs=8, verbose=5)(delayed(validate)(x, y_gr, svc, ind, i, acc_score) for i in set(ind))
        else:
            acc = Parallel(n_jobs=8, verbose=5)(delayed(validate)(x, y_gr, svc, ind, i, acc_score) for i in range(len(x)))

        if not np.mean(acc) <= best_acc:
            best_acc = np.mean(acc)
            best_std = np.std(acc)
            best_param = c
            best_model = svc
        time_el = time.time() - time_el
        print u'Param: C = {0} - Accuracy = {1} \u00B1 {2} - Best Acc. = {3} \u00B1 {4} - Time = {5}'.format(c, np.mean(acc), np.std(acc), best_acc, best_std, time_el)

    best_model.fit(x, y_gr)
    return best_model, best_param, best_acc, best_std


def reg_group_lopo(x, y, ind, rang, **kwargs):
    x = np.array([x[i] for i in range(len(x)) if rang[0] < y[i] < rang[1]])
    ind = np.array([ind[i] for i in range(len(ind)) if rang[0] < y[i] < rang[1]])
    y = np.array([y[i] for i in range(len(y)) if rang[0] < y[i] < rang[1]])

    # Parameter Search
    gamma = kwargs['gamma'] if 'gamma' in kwargs else np.array(range(1, 10))/10.0
    c_param = kwargs['c'] if 'c' in kwargs else np.array(range(1, 10))/10.0

    gamma, c_param = np.meshgrid(gamma, c_param)

    best_mae = np.nan
    best_std = 0
    best_param = []
    best_model = []
    for c, g in zip(gamma.reshape(-1), c_param.reshape(-1)):
        time_el = time.time()
        # Create estimator instance
        svr = svm.SVR(kernel='rbf', C=c, gamma=g)

        # Leave One Person Out (LOPO)
        if len(ind) > 0:
            mae = Parallel(n_jobs=8, verbose=5)(delayed(validate)(x, y, svr, ind, i, mae_score) for i in set(ind))
        else:
            mae = Parallel(n_jobs=8, verbose=5)(delayed(validate)(x, y, svr, ind, i, mae_score) for i in range(len(x)))

        if not np.mean(mae) >= best_mae:
            best_mae = np.mean(mae)
            best_std = np.std(mae)
            best_param = [c, g]
            best_model = svr
        time_el = time.time() - time_el
        print u'Params: (C = {0}, gamma = {1}) - MAE = {2} \u00B1 {3} - Best MAE = {4} \u00B1 {5} - Time = {6}'.format(c, g, np.mean(mae), np.std(mae), best_mae, best_std, time_el)

    best_model.fit(x, y)
    return best_model, best_param, best_mae, best_std

