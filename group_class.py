from __future__ import print_function
import time
from utils import *
from plotting import *
from joblib import Parallel, delayed
from sklearn import grid_search, svm
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier


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
        train_idx = i[0]
        test_idx = i[1]

    # Train Estimator
    estimator.fit(x[train_idx], y[train_idx])

    # Test Estimator
    return evel_func(estimator, x[test_idx], y[test_idx])


def group_class_lopo(x, y, ind, age_bins, **kwargs):
    # Get group labels
    y_gr = group_y(y, age_bins)

    # Parameter Search
    kernel = 'linear' if 'kernel' not in kwargs else kwargs['kernel']
    c_param = np.array(range(1, 100))/100.0 if 'c' not in kwargs else kwargs['c']
    gamma = [] if 'gamma' not in kwargs else kwargs['gamma']

    best_acc = np.nan
    best_std = 0
    best_param = []
    best_model = []
    n_folds = kwargs['n_folds'] if 'n_folds' in kwargs else 10
    n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
    verbose = kwargs['verbose'] if 'verbose' in kwargs else 0
    verboseprint = print if verbose else lambda *a, **k: None
    verbose -= 1 if verbose > 0 else 0

    if gamma:
        params = map(lambda x: x.reshape(-1), np.meshgrid(c_param, gamma))
        params = zip(params[0], params[1])
    else:
        params = c_param

    for p in params:
        if type(p) is tuple:
            c, g = p
        else:
            c = p
            g = 1

        time_el = time.time()
        # Create estimator instance
        svc = svm.SVC(kernel=kernel, C=c, gamma=g)
        # rand_forest = RandomForestClassifier(oob_score=True, n_estimators=c, n_jobs=n_jobs)

        # Leave One Person Out (LOPO)
        if len(ind) > 0:
            acc = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(validate)(x, y_gr, svc, ind, i, perf_score) for i in set(ind))
        else:
            kf = KFold(len(y_gr), n_folds=n_folds, shuffle=True)
            acc = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(validate)(x, y_gr, svc, ind, i, perf_score) for i in kf)

        if not np.mean(acc) <= best_acc:
            best_acc = np.mean(acc)
            best_std = np.std(acc)
            best_param = {'c': c, 'gamma': g}
            best_model = svc
        time_el = time.time() - time_el
        verboseprint(u'Param: {0} - Score = {1} \u00B1 {2} - Best Score = {3} \u00B1 {4} - Time = {5}'.format(p, np.mean(acc), np.std(acc), best_acc, best_std, time_el))

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

    n_folds = kwargs['n_folds'] if 'n_folds' in kwargs else 10
    n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
    verbose = kwargs['verbose'] if 'verbose' in kwargs else 0
    verboseprint = print if verbose else lambda *a, **k: None
    verbose -= 1 if verbose > 0 else 0

    for c, g in zip(gamma.reshape(-1), c_param.reshape(-1)):
        time_el = time.time()
        # Create estimator instance
        svr = svm.SVR(kernel='rbf', C=c, gamma=g)

        # Leave One Person Out (LOPO)
        if len(ind) > 0:
            mae = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(validate)(x, y, svr, ind, i, mae_score) for i in set(ind))
        else:
            kf = KFold(len(y), n_folds=n_folds, shuffle=True)
            mae = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(validate)(x, y, svr, ind, i, mae_score) for i in kf)

        if not np.mean(mae) >= best_mae:
            best_mae = np.mean(mae)
            best_std = np.std(mae)
            best_param = [c, g]
            best_model = svr
        time_el = time.time() - time_el
        verboseprint(u'Params: (C = {0}, gamma = {1}) - MAE = {2} \u00B1 {3} - Best MAE = {4} \u00B1 {5} - Time = {6}'.format(c, g, np.mean(mae), np.std(mae), best_mae, best_std, time_el))

    best_model.fit(x, y)
    return best_model, best_param, best_mae, best_std

