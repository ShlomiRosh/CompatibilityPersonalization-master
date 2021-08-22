import csv
import os.path
import shutil

import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from Models import Model, evaluate_params
from ExperimentSettings import get_experiment_parameters
import AnalyseResults
import random


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

    dataset_name = 'assistment'
    # dataset_name = 'citizen_science'
    # dataset_name = 'mooc'

    # model settings
    models_to_test = {
        'no hist': [1, 1, 0, 0],
        'm1': [0, 0, 1, 1],
        'm2': [0, 1, 1, 0],
        'm3': [0, 1, 1, 1],
        'm4': [1, 0, 0, 1],
        'm5': [1, 0, 1, 1],
        'm6': [1, 1, 0, 1],
        'm7': [1, 1, 1, 0],
        'm8': [1, 1, 1, 1],
    }

    # experiment settings
    chrono_split = True
    timestamp_split = False
    predetermined_timestamps = True
    keep_train_test_ratio = True
    min_subset_size = 5
    autotune_hyperparams = True
    autotune_autc = False
    normalize_numeric_features = False
    balance_histories = False

    # output settings
    overwrite_result_folder = True
    reset_cache = False
    only_test = False
    make_tradeoff_plots = True
    show_tradeoff_plots = True
    plot_confusion = False
    verbose = False

    dataset_dir = 'datasets/%s' % dataset_name
    result_dir = 'result'

    target_col, original_categ_cols, user_cols, skip_cols, hists_already_determined, df_max_size, train_frac, \
    valid_frac, h1_frac, h2_len, seeds, inner_seeds, weights_num, weights_range, model_params, min_hist_len, \
    max_hist_len, metrics, min_hist_len_to_test = get_experiment_parameters(dataset_name)
    test_frac = 1 - (train_frac + valid_frac)
    model_type = model_params['name']
    params = model_params['params']
    if not isinstance(next(iter(params.values())), list):
        autotune_hyperparams = False
    chosen_params = None
    if not autotune_hyperparams:
        chosen_params = params
    if timestamp_split:
        # if predetermined_timestamps and os.path.exists(timestamps_path):
        if predetermined_timestamps:
            timestamps_path = '%s/timestamp analysis/timestamp_splits.csv' % dataset_dir
            print('SPLIT BY TIMESTAMPS CROSS-VALIDATION MODE')
            seed_timestamps = pd.read_csv(timestamps_path)['timestamp']
            seeds = range(len(seed_timestamps))
        else:
            seed_timestamps = None

    # default settings
    diss_weights = list(np.linspace(0, 1, weights_num))
    model_names = list(models_to_test.keys())
    no_compat_equality_groups = [['no hist', 'm4', 'm6'], ['m1', 'm2', 'm3'], ['m5', 'm7', 'm8']]
    no_compat_equality_groups_per_model = {}
    for group in no_compat_equality_groups:
        for member in group:
            no_compat_equality_groups_per_model[member] = group

    # skip cols
    user_cols_not_skipped = []
    for user_col in user_cols:
        if user_col not in skip_cols:
            user_cols_not_skipped.append(user_col)
    original_categs_not_skipped = []
    for categ in original_categ_cols:
        if categ not in skip_cols:
            original_categs_not_skipped.append(categ)
    user_cols = user_cols_not_skipped
    original_categ_cols = original_categs_not_skipped

    # create results dir
    dataset_path = '%s/%s.csv' % (dataset_dir, dataset_name)
    if overwrite_result_folder and os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        with open('%s/parameters.csv' % result_dir, 'w', newline='') as file_out:
            writer = csv.writer(file_out)
            writer.writerow(['train_frac', 'valid_frac', 'dataset_max_size', 'h1_frac', 'h2_len', 'seeds',
                             'inner_seeds', 'weights_num', 'weights_range', 'min_hist_len', 'max_hist_len',
                             'chrono_split', 'timestamp_split', 'balance_histories', 'skip_cols', 'model_type',
                             'params'])
            writer.writerow(
                [train_frac, valid_frac, df_max_size, h1_frac, h2_len, len(seeds), len(inner_seeds),
                 weights_num, str(weights_range), min_hist_len, max_hist_len, chrono_split, timestamp_split,
                 balance_histories, str(skip_cols), model_type, params])
    header = ['user', 'len', 'seed', 'inner_seed', 'h1_acc', 'weight']
    for model_name in model_names:
        header.extend(['%s x' % model_name, '%s y' % model_name])

    # run whole experiment for each user column selection
    for user_col in user_cols:
        print('user column = %s' % user_col)
        done_by_seed = {}

        # create all folders
        result_type_dir = '%s/%s' % (result_dir, user_col)
        if not os.path.exists(result_type_dir):
            for metric in metrics:
                os.makedirs('%s/%s' % (result_type_dir, metric))
                for subset in ['train', 'valid', 'test']:
                    with open('%s/%s/%s_log.csv' % (result_type_dir, metric, subset), 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(header)

        else:  # load what is already done
            done_by_seed = {}
            df_done = pd.read_csv('%s/%s/test_log.csv' % (result_type_dir, metrics[-1]))
            groups_by_seed = df_done.groupby('seed')
            for seed, seed_group in groups_by_seed:
                done_by_inner_seed = {}
                done_by_seed[seed] = done_by_inner_seed
                groups_by_inner_seed = seed_group.groupby('inner_seed')
                for inner_seed, inner_seed_group in groups_by_inner_seed:
                    done_by_inner_seed[inner_seed] = len(pd.unique(inner_seed_group['user']))
            del df_done

        cache_dir = '%s/caches/%s skip_%s max_len_%d min_hist_%d max_hist_%d balance_%s' % (
            dataset_dir, user_col, len(skip_cols), df_max_size, min_hist_len, max_hist_len, balance_histories)
        if reset_cache and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        safe_make_dir(cache_dir)

        all_seeds_in_cache = True
        if balance_histories:
            for seed in seeds:
                if not os.path.exists('%s/%d.csv' % (cache_dir, seed)):
                    all_seeds_in_cache = False
                    break
        else:
            if not os.path.exists('%s/0.csv' % cache_dir):
                all_seeds_in_cache = False

        print('loading %s dataset...' % dataset_name)
        if not all_seeds_in_cache:
            categ_cols = original_categ_cols.copy()
            try:  # dont one hot encode the user_col
                categ_cols.remove(user_col)
            except ValueError:
                pass

            # load data
            dataset_full = pd.read_csv(dataset_path).drop(columns=skip_cols)
            if not timestamp_split and 'timestamp' in dataset_full.columns:
                dataset_full = dataset_full.drop(columns='timestamp')
            if df_max_size > 1:
                dataset_full = dataset_full[:df_max_size]
            elif df_max_size > 0:  # is a fraction
                dataset_full = dataset_full[:int(len(dataset_full) * df_max_size)]

            print('one-hot encoding the data... ')
            col_groups_dict = {}
            categs_unique_values = dataset_full[categ_cols].nunique()
            i = 0
            for col in dataset_full.columns:
                if col in [user_col, target_col]:
                    continue
                unique_count = 1
                if col in categ_cols:
                    unique_count = categs_unique_values[col]
                col_groups_dict[col] = range(i, i + unique_count)
                i = i + unique_count
            dataset_full = ce.OneHotEncoder(cols=categ_cols, use_cat_names=True).fit_transform(dataset_full)

            if hists_already_determined:  # todo: handle multiple seeds when balancing
                dataset_full.to_csv('%s/0.csv' % cache_dir, index=False)
                if not os.path.exists('%s/all_columns.csv' % cache_dir):
                    pd.DataFrame(columns=list(dataset_full.drop(columns=[user_col]).columns)).to_csv(
                        '%s/all_columns.csv' % cache_dir, index=False)
                del dataset_full
            else:
                print('sorting histories...')
                groups_by_user = dataset_full.groupby(user_col, sort=False)
                dataset_full = dataset_full.drop(columns=[user_col])
                all_columns = list(dataset_full.columns)
                if not os.path.exists('%s/all_columns.csv' % cache_dir):
                    pd.DataFrame(columns=all_columns).to_csv('%s/all_columns.csv' % cache_dir, index=False)
                del dataset_full

                # get user histories
                for seed in seeds:
                    if not os.path.exists('%s/%d.csv' % (cache_dir, seed)):
                        hists = {}
                        for user_id in groups_by_user.groups.keys():
                            hist = groups_by_user.get_group(user_id).drop(columns=[user_col])
                            if len(hist) < min_hist_len:
                                continue
                            if balance_histories:
                                target_groups = hist.groupby(target_col)
                                if len(target_groups) == 1:  # only one target label present in history: skip
                                    continue
                                hist = target_groups.apply(
                                    lambda x: x.sample(target_groups.size().min(), random_state=seed))
                                hist.index = hist.index.droplevel(0)
                            hists[user_id] = hist
                        sorted_hists = [[k, v] for k, v in reversed(sorted(hists.items(), key=lambda n: len(n[1])))]
                        seed_df = pd.DataFrame(columns=[user_col] + all_columns, dtype=np.int64)
                        for user_id, hist in sorted_hists:
                            hist[user_col] = [user_id] * len(hist)
                            seed_df = seed_df.append(hist, sort=False)
                        seed_df.to_csv('%s/0.csv' % cache_dir, index=False)
                    if not balance_histories:
                        break
                del groups_by_user
                del hists
        # end of making seed caches

        print("determine experiment's users...")
        min_max_col_values = pd.read_csv('%s/all_columns.csv' % cache_dir, dtype=np.int64)
        if timestamp_split:
            min_max_col_values = min_max_col_values.drop(columns='timestamp')
        all_columns = min_max_col_values.columns

        dataset = pd.read_csv('%s/0.csv' % cache_dir)

        if timestamp_split:
            if seed_timestamps is None:
                timestamp_min = dataset['timestamp'].min()
                timestamp_max = dataset['timestamp'].max()
                timestamp_range = timestamp_max - timestamp_min
                timestamp_h1_end = int(timestamp_min + timestamp_range * h1_frac)
                timestamp_valid_start = int(timestamp_min + timestamp_range * train_frac)
                timestamp_test_start = int(timestamp_valid_start + timestamp_range * valid_frac)
            else:
                hist_valid_fracs = np.linspace(1 - valid_frac, valid_frac, len(inner_seeds))

        groups_by_user = dataset.groupby(user_col, sort=False)
        hists_by_user = {}
        hist_train_ranges = {}
        curr_h2_len = 0
        num_users_to_test = 0
        user_ids = []
        for user_id, hist in groups_by_user:
            user_ids.append(user_id)
            hist = hist.drop(columns=[user_col])
            if timestamp_split and seed_timestamps is None:
                # if no pre-selected timestamps, have to find which users to use
                timestamp_hist_min = hist['timestamp'].min()
                timestamp_hist_max = hist['timestamp'].max()
                skip_user = False
                for t1, t2 in [[timestamp_min, timestamp_valid_start],
                               [timestamp_valid_start, timestamp_test_start],
                               [timestamp_test_start, timestamp_max]]:
                    if sum((hist['timestamp'] >= t1) & (hist['timestamp'] < t2)) < min_subset_size:
                        skip_user = True
                        break
                if skip_user:
                    continue
                hist_train_len = sum(hist['timestamp'] < timestamp_valid_start)
            else:
                hist_train_len = len(hist) * train_frac
            if hists_already_determined or (min_hist_len <= hist_train_len and curr_h2_len + hist_train_len <= h2_len):
                if len(hist) >= min_hist_len_to_test:
                    num_users_to_test += 1
                if len(hist) > max_hist_len:
                    hist = hist[:max_hist_len]
                hists_by_user[user_id] = hist
                min_max_col_values = min_max_col_values.append(hist.apply(min_and_max), sort=False)

                if chrono_split:
                    hist_train_ranges[user_id] = [curr_h2_len, len(hist)]
                    curr_h2_len += len(hist)
                else:
                    hist_train_ranges[user_id] = [curr_h2_len, curr_h2_len + int(len(hist) * train_frac)]
                    curr_h2_len += int(len(hist) * train_frac)
                    if curr_h2_len + min_hist_len * train_frac > h2_len:
                        break
        del groups_by_user

        if not chrono_split:
            # set hist train ranges
            for user_id, hist_train_range in hist_train_ranges.items():
                hist_train_len = hist_train_range[1] - hist_train_range[0]
                range_vector = np.zeros(curr_h2_len)
                for i in range(hist_train_range[0], hist_train_range[1]):
                    range_vector[i] = 1
                hist_train_ranges[user_id] = [range_vector, hist_train_len]

        print('cols=%d data_len=%d h1_frac=%s users=%d diss_weights=%d model_type=%s auto_tune_params=%s' % (
            len(all_columns) - 1, curr_h2_len, h1_frac, len(hists_by_user), len(diss_weights), model_type,
            autotune_hyperparams))

        min_max_col_values = min_max_col_values.reset_index(drop=True)
        scaler, labelizer = MinMaxScaler(), LabelBinarizer()
        if normalize_numeric_features:
            scaler.fit(min_max_col_values.drop(columns=[target_col]), min_max_col_values[[target_col]])
        labelizer.fit(min_max_col_values[[target_col]])
        del min_max_col_values

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

    # end user type loop

    print('\ndone')
