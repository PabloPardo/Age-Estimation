from __future__ import print_function
from group_class import *


def train_hybrid_age_estimator(x, y, age_bins, **kwargs):
    """
    Hybrid Age Estimation method based on Hu Han et al. (Age Estimation from Face
    Images: Human vs. Machine Performance). It has two phases, first it classifies
    the instances between 'n_reg' number of age ranges and then train one regressor
    for each range. The function train and validates the method for different parameters
    and returns the best performing ones.

    :param x: Instances (BIF features + Shapes (68 landmarks))
    :param y: Age labels
    :param age_bins: List of age bins to determine the age ranges
    :param kwargs: Extra parameters, such as verbose level, number of jobs
                   and number of folds in the 10-fold cv
    :return: Returns the best performing parameters and models
    """

    verbose = kwargs['verbose'] if 'verbose' in kwargs else 0
    verboseprint = print if verbose else lambda *a, **k: None
    verbose -= 1 if verbose > 0 else 0

    n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
    n_folds = kwargs['n_folds'] if 'n_folds' in kwargs else 10

    # Create the range
    # y_range = (max(y) - min(y)) / float(n_reg)
    # age_bins = [min(y) + i*y_range for i in range(n_reg+1)]

    # ---------------------------------------------------------------------------------
    # CLASSIFICATION
    # ---------------------------------------------------------------------------------
    verboseprint('Start age group classification ...')
    t = time.time()
    model_class = {}
    param_class = {}
    score_class = {}
    std_class = {}

    # class_model, class_param, class_score, class_std = train_regressor_age_estimator(x=x,
    #                                                                                  y=y,
    #                                                                                  n_folds=n_folds,
    #                                                                                  n_jobs=n_jobs,
    #                                                                                  verbose=verbose,
    #                                                                                  c=[0.001, 0.01, 0.1, 1])
    model_class['full'], \
        param_class['full'], \
        score_class['full'], \
        std_class['full'] = group_class_lopo(x=x,
                                             y=y,
                                             ind=[],
                                             age_bins=age_bins,
                                             kernel='rbf',
                                             n_folds=n_folds,
                                             n_jobs=n_jobs,
                                             verbose=verbose,
                                             gamma=[0.005, 0.01, 0.05, 0.1],
                                             c=[0.5, 1, 1.5, 10])
    t = time.time() - t
    verboseprint(u'Best Score: {0} \u00B1 {1} - Best Param: {2} - Time: {3}'.format(score_class['full'],
                                                                                    std_class['full'],
                                                                                    param_class['full'], t))
    # for r in range(len(age_bins)-2):
    #     x_range = np.array([x[i] for i in range(len(y)) if age_bins[r] < y[i] <= age_bins[r+2]])
    #     y_range = np.array([y[i] for i in range(len(y)) if age_bins[r] < y[i] <= age_bins[r+2]])
    #     model_class[r], \
    #         param_class[r], \
    #         score_class[r], \
    #         std_class[r] = group_class_lopo(x=x_range,
    #                                         y=y_range,
    #                                         ind=[],
    #                                         age_bins=age_bins,
    #                                         kernel='rbf',
    #                                         n_folds=n_folds,
    #                                         n_jobs=n_jobs,
    #                                         verbose=verbose,
    #                                         gamma=[0.001, 0.01, 0.1, 1],
    #                                         c=[0.001, 0.01, 0.1, 1])

    t = time.time() - t
    # verboseprint(u'Best Score: {0} \u00B1 {1} - Best Param: {2} - Time: {3}'.format(class_score,
    #                                                                                 class_std, class_param, t))

    # ---------------------------------------------------------------------------------
    # REGRESSION
    # ---------------------------------------------------------------------------------
    model_reg = {}
    param_reg = {}
    mae_reg = {}
    std_reg = {}
    for r in range(len(age_bins)-1):
        verboseprint('Training [{0},{1}] group regressor ...'.format(age_bins[r], age_bins[r+1]))
        t = time.time()
        model_reg[r], param_reg[r], mae_reg[r], std_reg[r] = reg_group_lopo(x=x,
                                                                            y=y,
                                                                            ind=[],
                                                                            n_jobs=n_jobs,
                                                                            verbose=verbose,
                                                                            rang=[age_bins[r], age_bins[r+1]],
                                                                            gamma=[0.001, 0.01, 0.1, 1],
                                                                            c=[0.001, 0.01, 0.1, 1])
                                                                            # gamma=np.array(range(1, 100, 5))/100.0,
                                                                            # c=np.array(range(1, 100, 5))/100.0)
        t = time.time() - t
        verboseprint(u'Best MAE: {0} \u00B1 {1} - Best Params: {2} - Time: {3}'.format(mae_reg[r], std_reg[r], param_reg[r], t))

    classification = {'model': model_class, 'param': param_class, 'score': score_class, 'std': std_class}
    regression = {'model': model_reg, 'param': param_reg, 'score': mae_reg, 'std': std_reg}

    return classification, regression


def train_regressor_age_estimator(x, y, **kwargs):
    """
    Regression Age Estimation method. The function train and validates the method
    for different parameters and returns the best performing ones.

    :param x: Instances (BIF features + Shapes (68 landmarks))
    :param y: Age labels
    :param kwargs: Extra parameters, such as verbose level, number of jobs
                   and number of folds in the 10-fold cv
    :return: Returns the best performing parameters and models
    """

    verbose = kwargs['verbose'] if 'verbose' in kwargs else 0
    verboseprint = print if verbose else lambda *a, **k: None
    verbose -= 1 if verbose > 0 else 0

    n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
    n_folds = kwargs['n_folds'] if 'n_folds' in kwargs else 10

    # ---------------------------------------------------------------------------------
    # TRAIN AND VALIDATION
    # ---------------------------------------------------------------------------------
    verboseprint('Training Age Regressor ...')
    t = time.time()
    model, param, mae, std = reg_group_lopo(x=x,
                                            y=y,
                                            ind=[],
                                            n_jobs=n_jobs,
                                            n_folds=n_folds,
                                            verbose=verbose,
                                            rang=[min(y), max(y)],
                                            gamma=[0.001, 0.01, 0.1, 1],
                                            c=[0.001, 0.01, 0.1, 1])
                                            # gamma=np.array(range(1, 100, 5))/100.0,
                                            # c=np.array(range(1, 100, 5))/100.0)
    t = time.time() - t
    verboseprint(u'Best MAE: {0} \u00B1 {1} - Best Params: {2} - Time: {3}'.format(mae, std, param, t))

    return {'model': model, 'param': param, 'score': mae, 'std': std}


def train_classifier_age_estimator(x, y, **kwargs):
    """
    Classifier Age Estimation method. The function train and validates the method
    for different parameters and returns the best performing ones.

    :param x: Instances (BIF features + Shapes (68 landmarks))
    :param y: Age labels
    :param kwargs: Extra parameters, such as verbose level, number of jobs
                   and number of folds in the 10-fold cv
    :return: Returns the best performing parameters and models
    """

    verbose = kwargs['verbose'] if 'verbose' in kwargs else 0
    verboseprint = print if verbose else lambda *a, **k: None
    verbose -= 1 if verbose > 0 else 0

    n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
    n_folds = kwargs['n_folds'] if 'n_folds' in kwargs else 10

    # ---------------------------------------------------------------------------------
    # TRAIN AND VALIDATION
    # ---------------------------------------------------------------------------------
    verboseprint('Training Age Classifier ...')
    t = time.time()
    model, param, mae, std = group_class_lopo(x=x,
                                              y=y,
                                              ind=[],
                                              age_bins=range(101),
                                              n_jobs=n_jobs,
                                              n_folds=n_folds,
                                              verbose=verbose,
                                              kernel='rbf',
                                              rang=[min(y), max(y) + 1],
                                              gamma=[0.001, 0.01, 0.1, 1],
                                              c=np.array(range(10, 20))/1000.0)
    t = time.time() - t
    verboseprint(u'Best MAE: {0} \u00B1 {1} - Best Params: {2} - Time: {3}'.format(mae, std, param, t))

    return {'model': model, 'param': param, 'score': mae, 'std': std}


def test_hybrid_age_estimator(x, y, age_bins, overlap_margin, class_model, reg_models):
    """
    Test the Hybrid Age Estimator, first by classifying into age groups and then
    merging the result of the specialized regressors.

    :param x: Test instances
    :param age_bins: List of age bins to determine the age ranges
    :param overlap_margin: Overlap between the contiguous range regressors.
    :param class_model: Trained model for the classifier
    :param reg_models: Trained model for the regressors

    :return:Predictions
    """
    # Predict with the classifier
    pred_full_group = class_model['full'].predict(x)
    # pred_group = group_y(y, age_bins)
    # Create the range
    # print(acc_score_(pred_full_group, group_y(y, age_bins)))
    # print(perf_score_(pred_full_group, group_y(y, age_bins)))

    pred_group = pred_full_group
    # pred_group = []
    # for i in range(len(x)):
    #     aux = [pred_full_group[i]]
    #     for j in range(len(age_bins)-2):
    #         if j <= pred_full_group[i] <= j+1:
    #             aux2 = class_model[j].predict(x[i])
    #             aux.append(j + aux2[0])
    #
    #     # Find the most repeated answer of the three classifiers
    #     max_rep = np.nan
    #     for j in set(aux):
    #         if not max_rep >= aux.count(j):
    #             max_rep = aux.count(j)
    #             prediction = j
    #     pred_group.append(prediction)
    #
    #
    # print(acc_score_(pred_group, group_y(y, age_bins)))
    # print(perf_score_(pred_group, group_y(y, age_bins)))

    # Predict with the regressor and join the predictions
    pred_y = []
    for i in range(len(x)):
        for j in range(len(age_bins)-1):
            if pred_group[i] == j:
                y_ = reg_models[j].predict(x[i])

                if age_bins[j] + overlap_margin > y_ and j >= 1:
                    y_2 = reg_models[j-1].predict(x[i])
                    y_ = [(y_[0] + y_2[0]) / 2]

                elif age_bins[j+1] + 1 - overlap_margin < y_ and j+1 <= len(age_bins):
                    y_2 = reg_models[j+1].predict(x[i])
                    y_ = [(y_[0] + y_2[0]) / 2]

                pred_y.append(y_[0])
                break

    return pred_y


def test_regressor_age_estimator(x, reg_model):
    """
    Test the Hybrid Age Estimator, first by classifying into age groups and then
    merging the result of the specialized regressors.

    :param x: Test instances
    :param reg_model: Trained model for the regressor

    :return:Predictions
    """
    return reg_model.predict(x)


def test_classifier_age_estimator(x, class_model):
    """
    Test the Hybrid Age Estimator, first by classifying into age groups and then
    merging the result of the specialized regressors.

    :param x: Test instances
    :param y: Test Labels
    :param class_model: Trained model for the classifier

    :return:Predictions
    """
    return class_model.predict(x)