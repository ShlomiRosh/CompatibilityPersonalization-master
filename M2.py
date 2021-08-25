import numpy as np
from sklearn.metrics import auc, roc_auc_score


class Model:
    def __init__(self, clf, model_name='model', old_model=None, diss_weight=None, subset_weights=None,
                 hist_range=None, params=None):
        self.model_name = model_name
        self.clf = clf
        self.old_model = old_model
        self.diss_weight = diss_weight
        self.subset_weights = subset_weights
        self.hist_range = hist_range
        self.params = params.copy()

        # self.base_params = {}
        # for key, value in list(self.params.items()):
        #     if '_base_' in key:
        #         del self.params[key]
        #         self.base_params[key[6:]] = value  # remove the '_base_' string
        #
        # if model_type == 'tree':
        #     self.predictor = DecisionTreeClassifier(random_state=1, **self.params)
        # elif model_type == 'ridge':
        #     self.predictor = Ridge(random_state=1, **self.params)
        # elif model_type == 'adaboost':
        self.predictor = clf.set_params(random_state=1, **self.params)
        # else:
        #     raise ValueError('invalid model type')

    def fit(self, x, y):
        self.predictor.fit(x, y, sample_weight=self.get_sample_weights(x, y))

    def get_sample_weights(self, x, y):
        if self.old_model is None:
            return None

        diss_weight = self.diss_weight
        general_loss, general_diss, hist_loss, hist_diss = self.subset_weights

        # getting old predictions
        old_predicted = np.round(self.old_model.predict(x))
        old_correct = np.equal(old_predicted, y).astype(int)

        if hist_loss == hist_diss == 0:  # no hist
            gen = (1 - diss_weight) * general_loss
            diss = diss_weight * old_correct * general_diss
        else:
            gen = (1 - diss_weight) * (general_loss + hist_loss * self.hist_range)
            diss = diss_weight * old_correct * (general_diss + hist_diss * self.hist_range)
            if diss.sum() == 0:  # h1 is incorrect on all history samples
                diss = diss_weight * (general_diss + hist_diss * self.hist_range)
        sample_weight = gen + diss
        return sample_weight

    def predict(self, x):
        return self.predictor.predict(x)

    def score(self, x, y, metric):
        new_predicted = self.predict(x)
        new_correct = np.equal(new_predicted, y).astype(int)
        if metric == 'acc':
            performance = np.mean(new_correct)
        elif metric == 'auc':
            y_true = y
            y_pred = self.predictor.predict_proba(x)
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            else:  # todo: doesnt work when label=1 is the first label
                y_pred = 1 - y_pred.reshape(-1)
            labels = np.unique(y)
            if len(labels) == 1:  # to avoid error when calculating auc
                y_true = y_true.copy()
                y_pred = y_pred.copy()
                y_true[-1] = 1 - y_true[-1]
                y_pred[-1] = 1 - y_pred[-1]
            performance = roc_auc_score(y_true, y_pred)
        if self.old_model is None:  # testing pre-update model
            return {'y': performance, 'predicted': new_predicted}
        old_correct = np.equal(self.old_model.predictor.predict(x), y).astype(int)
        sum_old_correct = np.sum(old_correct)

        if sum_old_correct == 0:
            compatibility = 1  # no errors can be new - dissonance can only be 0
        else:
            compatibility = np.sum(old_correct * new_correct) / np.sum(old_correct)
        return {'x': compatibility, 'y': performance, 'predicted': new_predicted}


def evaluate_params(model_type, train_x, train_y, valid_x, valid_y, metric, candidate_params, subset_weights=None,
                    old_model=None, hist_range=None, weights_num=5, get_autc=True, verbose=False):
    scores, evaluated_params = [], []
    evaluate_params_step(model_type, train_x, train_y, valid_x, valid_y, metric, candidate_params.copy(),
                         subset_weights, old_model, hist_range, weights_num, {}, scores, evaluated_params,
                         get_autc=get_autc, verbose=verbose)
    return scores, evaluated_params


def evaluate_params_step(model_type, train_x, train_y, valid_x, valid_y, metric, candidate_params, subset_weights,
                         old_model, hist_range, weights_num, params, scores, evaluated_params_list,
                         step=1, get_autc=True, verbose=False):
    if candidate_params:  # continue recursion
        param_name, param_values = next(iter(candidate_params.items()))
        del candidate_params[param_name]
        for param_value in param_values:
            new_params = params.copy()
            new_params[param_name] = param_value
            result = evaluate_params_step(
                model_type, train_x, train_y, valid_x, valid_y, metric, candidate_params.copy(), subset_weights,
                old_model, hist_range, weights_num, new_params, step + 1, evaluated_params_list,
                get_autc=get_autc, verbose=verbose)
            score, evaluated_params = result
            scores.append(score)
            evaluated_params_list.append(evaluated_params)
            if verbose:
                prefix = '\t' * step
                print('%sscore=%.5f params=%s' % (prefix, score, evaluated_params))

    else:  # final recursion step
        if subset_weights is None:  # pre-update model
            h1 = Model(model_type, params=params)
            h1.fit(train_x, train_y)
            return h1.score(valid_x, valid_y, metric)['y'], params
        else:  # baseline or personalized model
            if get_autc:
                coms, pers = [], []
                for weight in np.linspace(0, 1, weights_num):
                    model = Model(model_type, '', old_model, weight, subset_weights, hist_range, params)
                    model.fit(train_x, train_y)
                    result = model.score(valid_x, valid_y, metric)
                    coms.append(result['x'])
                    pers.append(result['y'])
                for i in range(1, len(coms)):  # make coms monotonically increasing for auc computation
                    if coms[i] < coms[i - 1]:
                        coms[i] = coms[i - 1]
                return auc([0] + coms, [pers[0]] + pers), params  # correct leftwards for fairness
            else:
                model = Model(model_type, '', old_model, 0, subset_weights, hist_range, params)
                model.fit(train_x, train_y)
                return model.score(valid_x, valid_y, metric)['y'], params
