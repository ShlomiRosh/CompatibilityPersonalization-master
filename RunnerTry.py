import csv
import json
import os.path
import random
from time import time

import numpy as np
import pandas as pd

import AnalyseResults
from Models import Model, evaluate_params


class ModuleTimer:
    def __init__(self, iterations):
        self.iterations = iterations
        self.curr_iteration = 0
        self.start_time = 0
        self.avg_runtime = 0
        self.eta = 0

    def start_iteration(self):
        self.start_time = int(round(time() * 1000))

    def end_iteration(self):
        runtime = (round(time() * 1000) - self.start_time) / 1000
        self.curr_iteration += 1
        self.avg_runtime = (self.avg_runtime * (self.curr_iteration - 1) + runtime) / self.curr_iteration
        self.eta = (self.iterations - self.curr_iteration) * self.avg_runtime
        return runtime


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def min_and_max(x):
    return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])


def get_time_string(time_in_seconds):
    eta_string = '%.1f(secs)' % (time_in_seconds % 60)
    if time_in_seconds >= 60:
        time_in_seconds /= 60
        eta_string = '%d(mins) %s' % (time_in_seconds % 60, eta_string)
        if time_in_seconds >= 60:
            time_in_seconds /= 60
            eta_string = '%d(hours) %s' % (time_in_seconds % 24, eta_string)
            if time_in_seconds >= 24:
                time_in_seconds /= 24
                eta_string = '%d(days) %s' % (time_in_seconds, eta_string)
    return eta_string


def log_progress(runtime, mod_str, verbose=True):
    runtime_string = get_time_string(runtime)
    eta = get_time_string(sum(timer.eta for timer in timers))
    iteration = sum([timer.curr_iteration for timer in timers])
    progress_row = '%d/%d\tmod=%s \tseed=%d/%d \tinner_seed=%d/%d \tuser=%d/%d \ttime=%s \tETA=%s' % \
                   (iteration, iterations, mod_str, seed_idx + 1, len(seeds), inner_seed_idx + 1,
                    len(inner_seeds), user_count, num_users_to_test, runtime_string, eta)
    with open('%s/progress_log.txt' % result_type_dir, 'a') as file:
        file.write('%s\n' % progress_row)
    if verbose:
        print(progress_row)
    pass


if __name__ == "__main__":

    fileObject = open("data.json", "r")
    jsonContent = fileObject.read()
    aList = json.loads(jsonContent)
    seeds = aList['seeds']
    inner_seeds = aList['inner_seeds']
    num_users_to_test = aList['num_users_to_test']
    autotune_hyperparams = aList['autotune_hyperparams']
    done_by_seed = aList['done_by_seed']
    hists_by_user = aList['hists_by_user']
    timestamp_split = aList['timestamp_split']
    seed_timestamps = aList['seed_timestamps']
    chrono_split = aList['chrono_split']
    chrono_split = aList['chrono_split']
    print('\nstart experiment!')

    timer_evaluating_params = ModuleTimer(len(seeds) * len(inner_seeds) * num_users_to_test)
    timer_validation_results = ModuleTimer(len(seeds) * len(inner_seeds) * num_users_to_test)
    timer_test_results = ModuleTimer(len(seeds) * num_users_to_test)
    timers = [timer_evaluating_params, timer_validation_results, timer_test_results]
    iterations = sum([timer.iterations for timer in timers])

    params_list = None

    loop_modes = [True, False]
    if not autotune_hyperparams:  # dont evaluate hyper-param
        loop_modes = [False]

    # todo: OUTER FOLDS LOOP
    for seed_idx, seed in enumerate(seeds):

        if seed in done_by_seed:  # check if seed was already done
            done_by_inner_seed = done_by_seed[seed]
            seed_is_done = len(done_by_inner_seed) == len(inner_seeds) and all(
                [done_users == len(hists_by_user) for i, done_users in done_by_inner_seed.items()])
        else:
            done_by_inner_seed = {}
            seed_is_done = False
        if seed_is_done:
            timer_evaluating_params.curr_iteration += len(inner_seeds) * len(hists_by_user)
            timer_validation_results.curr_iteration += len(inner_seeds) * len(hists_by_user)
            timer_test_results.curr_iteration += len(hists_by_user)
            continue

        if timestamp_split and seed_timestamps is not None:
            timestamp_test_start = seed_timestamps[seed_idx]

        # split the test sets
        hists_seed_by_user = {}
        hist_train_and_valid_ranges = {}
        h2_train_and_valid = pd.DataFrame(columns=all_columns, dtype=np.float32)
        for user_idx, item in enumerate(hists_by_user.items()):
            user_id, hist = item
            if chrono_split:  # time series nested cross-validation
                if timestamp_split:
                    hist_train_and_valid = hist.loc[hist['timestamp'] < timestamp_test_start]
                    hist_test = hist.loc[hist['timestamp'] >= timestamp_test_start].drop(columns='timestamp')
                    if keep_train_test_ratio:
                        max_hist_test_len = int(len(hist) * test_frac)
                        hist_test = hist_test[:min(len(hist_test), max_hist_test_len)]
                else:
                    valid_len = int(len(hist) * valid_frac)
                    test_len = int(len(hist) * test_frac)
                    min_idx = 3 * valid_len  # |train set| >= 2|valid set|
                    delta = len(hist) - test_len - min_idx  # space between min_idx and test_start_idx
                    delta_frac = list(np.linspace(1, 0, len(seeds)))
                    random.seed(user_idx)
                    random.shuffle(delta_frac)
                    test_start_idx = min_idx + int(delta * delta_frac[seed])
                    hist_train_and_valid = hist.iloc[0: test_start_idx]
                    hist_test = hist.iloc[test_start_idx: test_start_idx + test_len + 1]
            else:
                hist_train_and_valid = hist.sample(n=int(len(hist) * (train_frac + valid_frac)),
                                                   random_state=seed)
                hist_test = hist.drop(hist_train_and_valid.index).reset_index(drop=True)

            hist_train_and_valid_ranges[user_id] = [len(h2_train_and_valid),
                                                    len(h2_train_and_valid) + len(hist_train_and_valid)]
            h2_train_and_valid = h2_train_and_valid.append(hist_train_and_valid, ignore_index=True,
                                                           sort=False)

            if normalize_numeric_features:
                hist_test_x = scaler.transform(hist_test.drop(columns=[target_col]))
            else:
                hist_test_x = hist_test.drop(columns=[target_col])
            hist_test_y = labelizer.transform(hist_test[[target_col]]).ravel()
            hists_seed_by_user[user_id] = [hist_train_and_valid, hist_test_x, hist_test_y]

        h2_train_and_valid_x = h2_train_and_valid.drop(columns=[target_col])
        if normalize_numeric_features:
            h2_train_and_valid_x = scaler.transform(h2_train_and_valid_x)
        h2_train_and_valid_y = labelizer.transform(h2_train_and_valid[[target_col]]).ravel()

        # todo: INNER FOLDS LOOP
        for evaluating_params in loop_modes:

            if evaluating_params:
                scores_per_user = {u: {m: [] for m in model_names} for u in user_ids}
            else:
                best_params_per_user = {u: {m: params_list[np.argmax(np.mean(scores_per_user[u][m], axis=0))]
                                            for m in model_names} for u in user_ids}

            for inner_seed_idx, inner_seed in enumerate(inner_seeds):

                if not evaluating_params:
                    if inner_seed in done_by_inner_seed:  # check if inner seed was already done
                        done_last_users = done_by_inner_seed[inner_seed]
                        inner_seed_is_done = done_last_users == len(hists_by_user)
                    else:
                        done_last_users = 0
                        inner_seed_is_done = False
                    if inner_seed_is_done:
                        timer_validation_results.curr_iteration += len(hists_by_user)
                        continue

                # split to train and validation sets
                hists_inner_seed_by_user = {}
                if h1_frac <= 1:  # if > 1 then simply take this number of samples
                    h1_train = pd.DataFrame(columns=all_columns, dtype=np.float32)
                h2_train = pd.DataFrame(columns=all_columns, dtype=np.float32)
                h2_valid = pd.DataFrame(columns=all_columns, dtype=np.float32)

                # todo: TRAIN-VALIDATION SPLITTING LOOP
                for user_idx, entry in enumerate(hists_seed_by_user.items()):
                    user_id, item = entry
                    hist_train_and_valid, hist_test_x, hist_test_y = item

                    if chrono_split:
                        if timestamp_split:
                            h = hist_train_and_valid
                            if seed_timestamps is None:  # does not support inner cross-validation
                                hist_train = h.loc[h['timestamp'] < timestamp_valid_start].drop(columns='timestamp')
                                hist_valid = h.loc[h['timestamp'] >= timestamp_valid_start].drop(
                                    columns='timestamp')
                            else:
                                hist_valid_len = int(len(h) * (hist_valid_fracs[inner_seed_idx]))
                                hist_train = h[:hist_valid_len].drop(columns='timestamp')
                                hist_valid = h[hist_valid_len:].drop(columns='timestamp')
                        else:
                            hist_len = hist_train_ranges[user_id][1]
                            valid_len = int(hist_len * valid_frac)
                            delta = len(
                                hist_train_and_valid) - 2 * valid_len  # space between min_idx and valid_start
                            delta_frac = list(np.linspace(1, 0, len(inner_seeds)))
                            random.seed(user_idx)
                            random.shuffle(delta_frac)
                            valid_start_idx = valid_len + int(delta * delta_frac[inner_seed])
                            hist_train = hist_train_and_valid.iloc[0: valid_start_idx]
                            # hist_valid = hist_train_and_valid.iloc[valid_start_idx: valid_start_idx + valid_len + 1]
                            hist_valid = hist_train_and_valid.iloc[valid_start_idx:]
                        hist_train_ranges[user_id][0] = [len(h2_train), len(h2_train) + len(hist_train)]
                    else:
                        hist_train_len = hist_train_ranges[user_id][1]
                        hist_train = hist_train_and_valid.sample(n=hist_train_len, random_state=inner_seed)
                        hist_valid = hist_train_and_valid.drop(hist_train.index)
                    if h1_frac <= 1:
                        if timestamp_split:
                            h = hist_train_and_valid
                            if seed_timestamps is None:
                                h1_hist_train = h.loc[h['timestamp'] <= timestamp_h1_end].drop(columns='timestamp')
                            else:
                                h1_hist_len = max(int(len(h) * h1_frac), 1)
                                h1_hist_train = h[:h1_hist_len].drop(columns='timestamp')
                        else:
                            h1_hist_train = hist_train[:max(int(len(hist_train_and_valid) * h1_frac), 1)]
                        h1_train = h1_train.append(h1_hist_train, ignore_index=True, sort=False)
                    h2_train = h2_train.append(hist_train, ignore_index=True, sort=False)
                    h2_valid = h2_valid.append(hist_valid, ignore_index=True, sort=False)
                    hists_inner_seed_by_user[user_id] = [hist_train, hist_valid, hist_test_x, hist_test_y]
                if h1_frac <= 1:
                    h1_train_x = h1_train.drop(columns=[target_col])
                h2_train_x = h2_train.drop(columns=[target_col])
                h2_valid_x = h2_valid.drop(columns=[target_col])
                if normalize_numeric_features:
                    if h1_frac <= 1:
                        h1_train_x = scaler.transform(h1_train_x)
                    h2_train_x = scaler.transform(h2_train_x)
                    h2_valid_x = scaler.transform(h2_valid_x)
                h2_train_y = labelizer.transform(h2_train[[target_col]]).ravel()
                h2_valid_y = labelizer.transform(h2_valid[[target_col]]).ravel()
                if h1_frac <= 1:
                    h1_train_y = labelizer.transform(h1_train[[target_col]]).ravel()
                else:
                    h1_train_x = h1_train.drop(columns=[target_col])
                    h1_train_y = labelizer.transform(h1_train[[target_col]]).ravel()

                tuning_x, tuning_y = h2_valid_x, h2_valid_y

                # train h1 and baseline
                if autotune_hyperparams:
                    if 'h1' not in model_params['forced_params_per_model']:
                        if verbose:
                            print('  h1:')
                        scores, evaluated_params = evaluate_params(
                            model_type, h1_train_x, h1_train_y, tuning_x, tuning_y, metrics[0], params,
                            get_autc=autotune_autc, verbose=verbose)
                        # scores_h1.append(scores)
                        if params_list is None:
                            params_list = evaluated_params
                    h1 = Model(model_type, 'h1', params=params_list[np.argmax(scores)])
                else:
                    h1 = Model(model_type, 'h1', params=chosen_params)
                h1.fit(h1_train_x, h1_train_y)

                user_count = 0

                # todo: USER LOOP
                for user_id, item in hists_inner_seed_by_user.items():
                    hist_train, hist_valid, hist_test_x, hist_test_y = item
                    if chrono_split:
                        hist_train_range = np.zeros(len(h2_train))
                        start_idx, end_idx = hist_train_ranges[user_id][0]
                        hist_train_range[start_idx:end_idx] = 1
                    else:
                        hist_train_range = hist_train_ranges[user_id][0]
                    hist_len = len(hist_train)

                    user_count += 1
                    if not evaluating_params:
                        if user_count <= done_last_users:
                            timer_validation_results.curr_iteration += 1
                            continue
                        timer_validation_results.start_iteration()
                    else:
                        timer_evaluating_params.start_iteration()

                    # prepare train and validation sets
                    if normalize_numeric_features:
                        hist_train_x = scaler.transform(hist_train.drop(columns=[target_col]))
                        hist_valid_x = scaler.transform(hist_valid.drop(columns=[target_col]))
                    else:
                        hist_train_x = hist_train.drop(columns=[target_col])
                        hist_valid_x = hist_valid.drop(columns=[target_col])
                    hist_train_y = labelizer.transform(hist_train[[target_col]]).ravel()
                    hist_valid_y = labelizer.transform(hist_valid[[target_col]]).ravel()

                    tuning_x, tuning_y = hist_valid_x, hist_valid_y

                    # train all models
                    if evaluating_params:
                        scores_per_model = {}
                        for model_name in model_names:
                            if verbose:
                                print('  %s:' % model_name)
                            if model_name not in model_params['forced_params_per_model']:
                                found = False
                                if not autotune_autc:  # look for best params to steal from other models
                                    for member in no_compat_equality_groups_per_model[model_name]:
                                        if member in scores_per_model:
                                            scores_per_model[model_name] = scores_per_model[member]
                                            found = True
                                            break
                                if not found:
                                    subset_weights = models_to_test[model_name]
                                    scores = evaluate_params(
                                        model_type, h2_train_x, h2_train_y, tuning_x, tuning_y, metrics[0],
                                        params, subset_weights, h1, hist_train_range,
                                        get_autc=autotune_autc, verbose=verbose)[0]
                                    scores_per_model[model_name] = scores
                                scores = scores_per_model[model_name]
                                scores_per_user[user_id][model_name].append(scores)
                    else:
                        if not only_test:
                            best_params_per_model = best_params_per_user[user_id]
                            models_by_weight = []
                            for weight_idx, weight in enumerate(diss_weights):
                                models = []
                                models_by_weight.append(models)
                                for model_name in model_names:
                                    subset_weights = models_to_test[model_name]
                                    best_params = best_params_per_model.get(model_name, chosen_params)
                                    model = Model(model_type, model_name, h1, weight, subset_weights,
                                                  hist_train_range, params=best_params)
                                    model.fit(h2_train_x, h2_train_y)
                                    models.append(model)

                            # test all models on validation set
                            rows_by_metric = []
                            for metric in metrics:
                                rows_by_subset = []
                                rows_by_metric.append(rows_by_subset)
                                subsets = ['train', 'valid']
                                for subset in subsets:
                                    x, y = eval('hist_%s_x' % subset), eval('hist_%s_y' % subset)
                                    rows = []
                                    rows_by_subset.append(rows)
                                    h1_y = h1.score(x, y, metric)['y']
                                    for weight_idx, weight in enumerate(diss_weights):
                                        models = models_by_weight[weight_idx]
                                        row = [user_id, hist_len, seed, inner_seed, h1_y, weight]
                                        for i, model in enumerate(models):
                                            result = model.score(x, y, metric)
                                            com, acc = result['x'], result['y']
                                            row.extend([com, acc])
                                        rows.append(row)

                            # write rows to all logs in one go to avoid discrepancies between logs
                            for metric_idx, metric in enumerate(metrics):
                                for subset_idx, subset in enumerate(subsets):
                                    with open('%s/%s/%s_log.csv' % (result_type_dir, metric, subset), 'a',
                                              newline='') as file:
                                        writer = csv.writer(file)
                                        for row in rows_by_metric[metric_idx][subset_idx]:
                                            writer.writerow(row)

                    # end iteration
                    if evaluating_params:
                        runtime = timer_evaluating_params.end_iteration()
                        mod_str = 'params'
                    else:
                        runtime = timer_validation_results.end_iteration()
                        mod_str = 'valid'
                    log_progress(runtime, mod_str)
                # end user loop
            # end inner folds loop
        # end train and validation loop

        # todo: FINAL TESTING OF MODELS
        user_count = 0
        for user_idx, entry in enumerate(hists_seed_by_user.items()):
            timer_test_results.start_iteration()
            user_count += 1
            user_id, item = entry

            hist_train_and_valid, hist_test_x, hist_test_y = item
            if chrono_split:
                hist_train_and_valid_range = np.zeros(len(h2_train_and_valid))
                start_idx, end_idx = hist_train_and_valid_ranges[user_id]
                hist_train_and_valid_range[start_idx:end_idx] = 1
            else:
                hist_train_and_valid_range = hist_train_and_valid_ranges[user_id]
            hist_len = len(hist_train_and_valid)

            if autotune_hyperparams:
                best_params_per_model = best_params_per_user[user_id]
            else:
                best_params_per_model = {}
            models_by_weight = []
            for weight_idx, weight in enumerate(diss_weights):
                models = []
                models_by_weight.append(models)
                for model_name in model_names:
                    subset_weights = models_to_test[model_name]
                    best_params = best_params_per_model.get(model_name, chosen_params)
                    model = Model(model_type, model_name, h1, weight, subset_weights,
                                  hist_train_and_valid_range, params=best_params)
                    model.fit(h2_train_and_valid_x, h2_train_and_valid_y)
                    models.append(model)

            # test all models on validation set
            rows_by_metric = []
            for metric in metrics:
                rows = []
                rows_by_metric.append(rows)
                h1_y = h1.score(hist_test_x, hist_test_y, metric)['y']
                for weight_idx, weight in enumerate(diss_weights):
                    models = models_by_weight[weight_idx]
                    row = [user_id, hist_len, seed, inner_seed, h1_y, weight]
                    for i, model in enumerate(models):
                        result = model.score(hist_test_x, hist_test_y, metric)
                        com, acc = result['x'], result['y']
                        row.extend([com, acc])
                    rows.append(row)

            # write rows to all logs in one go to avoid discrepancies between logs
            for metric_idx, metric in enumerate(metrics):
                with open('%s/%s/test_log.csv' % (result_type_dir, metric), 'a', newline='') as file:
                    writer = csv.writer(file)
                    for row in rows_by_metric[metric_idx]:
                        writer.writerow(row)

            # end iteration
            runtime = timer_test_results.end_iteration()
            mod_str = 'test'
            log_progress(runtime, mod_str)
    # end outer folds loop

    if make_tradeoff_plots:
        log_dir = '%s/%s' % (result_type_dir, metrics[0])
        if len(model_names):
            AnalyseResults.binarize_results_by_compat_values(log_dir, 'test', len(diss_weights) * 4,
                                                             print_progress=False)
            models_for_plotting = AnalyseResults.get_model_dict('jet')
            AnalyseResults.plot_results(log_dir, dataset_name, models_for_plotting, 'test_bins', True,
                                        show_tradeoff_plots=show_tradeoff_plots, diss_labels=False,
                                        performance_metric=metrics[0])
        else:  # only h1
            df = pd.read_csv('%s/test_log.csv' % log_dir)
            print(np.average(df['h1_acc'], weights=df['len']))

    print('\ndone')
