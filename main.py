from group_class import *
import numpy as np
import sys
import codecs
from sklearn.cross_validation import KFold

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

# Load data
dataset_name = 'HuPBA'
x, y, shapes, gender, ind = load_data(dataset_name=dataset_name, type_label='real')

# train_methodology = 'Leave One Person Out'
train_methodology = '10-Fold Cross Validation'

# Joint BIF features with shape features
x = np.concatenate((x, shapes), axis=1)

mae = []
cum = []

n_folds = 10
n_jobs = 8
verbose = 0
fold = 0
kf = KFold(len(x), n_folds=n_folds)
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
    print 'Start age group classification ...'
    t = time.time()
    age_bins = [16, 35, 60, 100]
    class_model, class_param, class_acc, class_std = group_class_lopo(x=x_tr,
                                                                      y=y_tr,
                                                                      ind=[],
                                                                      age_bins=age_bins,
                                                                      n_folds=n_folds,
                                                                      n_jobs=n_jobs,
                                                                      verbose=verbose,
                                                                      c=np.array(range(1, 200, 10))/1000.0)
    t = time.time() - t
    print u'Best Accuracy: {0} \u00B1 {1}\nBest Param: {2}\nTime: {3}'.format(class_acc, class_std, class_param, t)

    # Regression of each ot the three age groups
    print 'Start Youth Regression ...'
    t = time.time()
    reg_yg_model, reg_yg_param, reg_yg_mae, reg_yg_std = reg_group_lopo(x=x_tr,
                                                                        y=y_tr,
                                                                        ind=[],
                                                                        n_folds=n_folds,
                                                                        n_jobs=n_jobs,
                                                                        verbose=verbose,
                                                                        rang=[0, 21],
                                                                        gamma=np.array(range(1, 100, 5))/100.0,
                                                                        c=np.array(range(1, 100, 5))/100.0)
    t = time.time() - t
    print u'Best MAE: {0} \u00B1 {1}\nBest Params: {2}\nTime: {3}'.format(reg_yg_mae, reg_yg_std, reg_yg_param, t)

    print 'Start Young-Adult Regression ...'
    t = time.time()
    reg_ya_model, reg_ya_param, reg_ya_mae, reg_ya_std = reg_group_lopo(x=x_tr,
                                                                        y=y_tr,
                                                                        ind=[],
                                                                        n_folds=n_folds,
                                                                        n_jobs=n_jobs,
                                                                        verbose=verbose,
                                                                        rang=[11, 40],
                                                                        gamma=np.array(range(1, 100, 5))/100.0,
                                                                        c=np.array(range(1, 100, 5))/100.0)
    t = time.time() - t
    print u'Best MAE: {0} \u00B1 {1}\nBest Params: {2}\nTime: {3}'.format(reg_ya_mae, reg_ya_std, reg_ya_param, t)

    print 'Start Mid-Adult Regression ...'
    t = time.time()
    reg_ma_model, reg_ma_param, reg_ma_mae, reg_ma_std = reg_group_lopo(x=x_tr,
                                                                        y=y_tr,
                                                                        ind=[],
                                                                        n_folds=n_folds,
                                                                        n_jobs=n_jobs,
                                                                        verbose=verbose,
                                                                        rang=[30, 65],
                                                                        gamma=np.array(range(1, 100, 5))/100.0,
                                                                        c=np.array(range(1, 100, 5))/100.0)
    t = time.time() - t
    print u'Best MAE: {0} \u00B1 {1}\nBest Params: {2}\nTime: {3}'.format(reg_ma_mae, reg_ma_std, reg_ma_param, t)

    print 'Start Old Regression ...'
    t = time.time()
    reg_ol_model, reg_ol_param, reg_ol_mae, reg_ol_std = reg_group_lopo(x=x_tr,
                                                                        y=y_tr,
                                                                        ind=[],
                                                                        n_folds=n_folds,
                                                                        n_jobs=n_jobs,
                                                                        verbose=verbose,
                                                                        rang=[55, 100],
                                                                        gamma=np.array(range(1, 100, 5))/100.0,
                                                                        c=np.array(range(1, 100, 5))/100.0)
    t = time.time() - t
    print u'Best MAE: {0} \u00B1 {1}\nBest Params: {2}\nTime: {3}'.format(reg_ol_mae, reg_ol_std, reg_ol_param, t)

    # ---------------------------------------------------------------------------------
    # TEST
    # ---------------------------------------------------------------------------------
    pred_group = class_model.predict(x_ts)

    pred_y = []
    w = class_acc/100.0
    for j in range(len(x_ts)):
        if pred_group[j] == 0:
            aux = reg_yg_model.predict(x_ts[j])
            if aux[0] > 11:
                aux_ = reg_ma_model.predict(x_ts[j])
                aux = [(aux[0] + aux_[0]) / 2]
            pred_y.append(aux[0])

        elif pred_group[j] == 1:
            aux = reg_ya_model.predict(x_ts[j])
            if aux[0] < 21:
                aux_ = reg_yg_model.predict(x_ts[j])
                aux = [(aux[0] + aux_[0]) / 2]
            elif aux[0] > 30:
                aux_ = reg_ma_model.predict(x_ts[j])
                aux = [(aux[0] + aux_[0]) / 2]
            pred_y.append(aux[0])

        elif pred_group[j] == 2:
            aux = reg_ma_model.predict(x_ts[j])
            if aux[0] < 40:
                aux_ = reg_ya_model.predict(x_ts[j])
                aux = [(aux[0] + aux_[0]) / 2]
            elif aux[0] > 55:
                aux_ = reg_ol_model.predict(x_ts[j])
                aux = [(aux[0] + aux_[0]) / 2]
            pred_y.append(aux[0])

        elif pred_group[j] == 3:
            aux = reg_ol_model.predict(x_ts[j])
            if aux[0] < 65:
                aux_ = reg_ma_model.predict(x_ts[j])
                aux = [(aux[0] + aux_[0]) / 2]
            pred_y.append(aux[0])

        else:
            raise NameError('The predicted label - {0} - is not correct'.format(pred_group[j]))

    try:
        mae.append(mae_score_(y_ts, pred_y))
        cum.append(cum_score_(y_ts, pred_y))
    except Exception, e:
        print e
        print 'Testing Y: {0}\Predicted Y: {1}'.format(y_ts, pred_y)

    time_el = time.time() - time_el
    print u'\n\n-----------\nTest MAE: {0} - Test mean MAE: {1} \u00B1 {2} - time: {3}\n-----------'.format(mae[-1], np.mean(mae), np.std(mae), time_el)

cum_score(name='images/%s_cum_score.png' % dataset_name, cum_score=np.mean(cum, axis=0))
print u'\n\n-------- MAE = {0} \u00B1 {1} --------'.format(np.mean(mae), np.std(mae))