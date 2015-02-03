import sys
import codecs
import copy
from methods import *
from sklearn.cross_validation import KFold, StratifiedKFold

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

# Load data
dataset_name = 'HuPBA'
x, y, shapes, gender, ind = load_data(dataset_name=dataset_name, type_label='both')

# train_methodology = 'Leave One Person Out'
train_methodology = '10-Fold Cross Validation'

# Joint BIF features with shape features
x = np.concatenate((x, shapes), axis=1)
y_ = copy.deepcopy(y)

mae = {'real': [], 'apparent': []}
cum = {'real': [], 'apparent': []}

# Parameters
# age_bins = [0, 20, 35, 55, 101]
age_bins = [0, 25, 50,  101]
n_folds = 10
n_in_folds = 10
n_jobs = -1
overlap = 3
verbose = 1

for type_y in [0, 1]:
    y = y_[:, type_y]

    # y_range = (max(y) - min(y)) / float(n_reg)
    # age_bins = [min(y) + i*y_range for i in range(n_reg+1)]

    fold = 0
    kf = KFold(len(y), n_folds=n_folds, shuffle=True)
    for train_idx, test_idx in kf:
        time_el = time.time()
        print '------------------- {0} loop: {1}/{2} -------------------'.format(train_methodology, fold, n_folds)
        fold += 1

        x_tr = x[train_idx]
        y_tr = y[train_idx]

        x_ts = x[test_idx]
        y_ts = y[test_idx]

        # ---------------------------------------------------------------------------------
        # TRAIN AND VALIDATION
        # ---------------------------------------------------------------------------------

        # Classification within the different age groups
        classify, regress = train_hybrid_age_estimator(x=x_tr,
                                                       y=y_tr,
                                                       age_bins=age_bins,
                                                       n_jobs=n_jobs,
                                                       verbose=verbose,
                                                       n_folds=n_in_folds)

        # regress = train_regressor_age_estimator(x=x_tr,
        #                                         y=y_tr,
        #                                         n_jobs=n_jobs,
        #                                         verbose=verbose,
        #                                         n_folds=n_in_folds)

        # classify = train_classifier_age_estimator(x=x_tr,
        #                                           y=y_tr,
        #                                           n_jobs=n_jobs,
        #                                           verbose=verbose,
        #                                           n_folds=n_in_folds)

        # ---------------------------------------------------------------------------------
        # TEST
        # ---------------------------------------------------------------------------------

        pred_y = test_hybrid_age_estimator(x=x_ts,
                                           y=y_ts,
                                           age_bins=age_bins,
                                           overlap_margin=overlap,
                                           class_model=classify['model'],
                                           reg_models=regress['model'])


        # pred_y = test_regressor_age_estimator(x=x_ts,
        #                                       reg_model=regress['model'])

        # pred_y = test_classifier_age_estimator(x=x_ts,
        #                                        class_model=classify['model'])

        try:
            mae[mae.keys()[type_y]].append(mae_score_(y_ts, pred_y))
            cum[cum.keys()[type_y]].append(cum_score_(y_ts, pred_y))
        except Exception, e:
            print e
            print 'Testing Y: {0}\nPredicted Y: {1}'.format(y_ts, pred_y)

        time_el = time.time() - time_el
        print u'\n\n-----------\nTest MAE: {0} - Test mean MAE: {1} \u00B1 {2} - time: {3}\n-----------'.format(mae[mae.keys()[type_y]][-1], np.mean(mae[mae.keys()[type_y]]), np.std(mae[mae.keys()[type_y]]), time_el)

    print u'\n\n-------- MAE = {0} \u00B1 {1} --------'.format(np.mean(mae[mae.keys()[type_y]]), np.std(mae[mae.keys()[type_y]]))

cum_score('images/%s_hybrid_cum_score2.png' % dataset_name,
          np.mean(cum['real'], axis=0),
          np.mean(cum['apparent'], axis=0),
          legend=['Real', 'Apparent'])
print u'\nREAL AGE\n-------- MAE = {0} \u00B1 {1} --------'.format(np.mean(mae['real']), np.std(mae['real']))
print u'\nAPPARENT AGE\n-------- MAE = {0} \u00B1 {1} --------'.format(np.mean(mae['apparent']), np.std(mae['apparent']))