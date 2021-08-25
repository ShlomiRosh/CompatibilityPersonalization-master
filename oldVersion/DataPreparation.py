import csv
import os.path
import shutil

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

import ExperimentSettings as es
from ExperimentSettings import get_experiment_parameters


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def min_and_max(x):
    return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])


class DataPreparations:

    def __init__(self):
        self.__prepared_params = {}
        self.__run_prepartion()

    def get_experiment_parameters(self):
        return self.__prepared_params

    def __run_prepartion(self):
        dataset_name = es.dataset_name
        model_names = es.model_names
        # experiment settings
        chrono_split = True
        timestamp_split = False
        autotune_hyperparams = True

        balance_histories = False

        # output settings
        overwrite_result_folder = True
        reset_cache = False
        seed_timestamps = None
        dataset_dir = 'datasets/%s' % dataset_name
        result_dir = '../result'

        target_col, original_categ_cols, user_cols, skip_cols, hists_already_determined, df_max_size, train_frac, \
        valid_frac, h1_frac, h2_len, seeds, inner_seeds, weights_num, weights_range, model_params, min_hist_len, \
        max_hist_len, metrics, min_hist_len_to_test = get_experiment_parameters(dataset_name)
        model_type = model_params['name']
        params = model_params['params']
        if not isinstance(next(iter(params.values())), list):
            autotune_hyperparams = False

        #     TODO
        # if not autotune_hyperparams:
        #     chosen_params = params
        chosen_params = params

        # TODO
        # if timestamp_split:
        #     # if predetermined_timestamps and os.path.exists(timestamps_path):
        #     if predetermined_timestamps:
        #         timestamps_path = '%s/timestamp analysis/timestamp_splits.csv' % dataset_dir
        #         print('SPLIT BY TIMESTAMPS CROSS-VALIDATION MODE')
        #         seed_timestamps = pd.read_csv(timestamps_path)['timestamp']
        #         seeds = range(len(seed_timestamps))
        #     else:
        #         seed_timestamps = None

        # default settings
        diss_weights = list(np.linspace(0, 1, weights_num))

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
            # TODO
            # if timestamp_split:
            #     min_max_col_values = min_max_col_values.drop(columns='timestamp')
            all_columns = min_max_col_values.columns

            dataset = pd.read_csv('%s/0.csv' % cache_dir)

            # TODO
            # if timestamp_split:
            #     if seed_timestamps is None:
            #         timestamp_min = dataset['timestamp'].min()
            #         timestamp_max = dataset['timestamp'].max()
            #         timestamp_range = timestamp_max - timestamp_min
            #         timestamp_h1_end = int(timestamp_min + timestamp_range * h1_frac)
            #         timestamp_valid_start = int(timestamp_min + timestamp_range * train_frac)
            #         timestamp_test_start = int(timestamp_valid_start + timestamp_range * valid_frac)
            #     else:
            #         hist_valid_fracs = np.linspace(1 - valid_frac, valid_frac, len(inner_seeds))

            groups_by_user = dataset.groupby(user_col, sort=False)
            hists_by_user = {}
            hist_train_ranges = {}
            curr_h2_len = 0
            num_users_to_test = 0
            user_ids = []
            for user_id, hist in groups_by_user:
                user_ids.append(user_id)
                hist = hist.drop(columns=[user_col])
                # TODO
                # if timestamp_split and seed_timestamps is None:
                #     # if no pre-selected timestamps, have to find which users to use
                #     timestamp_hist_min = hist['timestamp'].min()
                #     timestamp_hist_max = hist['timestamp'].max()
                #     skip_user = False
                #     for t1, t2 in [[timestamp_min, timestamp_valid_start],
                #                    [timestamp_valid_start, timestamp_test_start],
                #                    [timestamp_test_start, timestamp_max]]:
                #         if sum((hist['timestamp'] >= t1) & (hist['timestamp'] < t2)) < min_subset_size:
                #             skip_user = True
                #             break
                #     if skip_user:
                #         continue
                #     hist_train_len = sum(hist['timestamp'] < timestamp_valid_start)
                # else:
                hist_train_len = len(hist) * train_frac
                if hists_already_determined or (
                        min_hist_len <= hist_train_len and curr_h2_len + hist_train_len <= h2_len):
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
            # TODO
            # if normalize_numeric_features:
            #     scaler.fit(min_max_col_values.drop(columns=[target_col]), min_max_col_values[[target_col]])
            labelizer.fit(min_max_col_values[[target_col]])
            del min_max_col_values

        self.__prepared_params = {
            'seeds': seeds, 'inner_seeds': inner_seeds, 'num_users_to_test': num_users_to_test,
            'autotune_hyperparams': autotune_hyperparams, 'done_by_seed': done_by_seed,
            'hists_by_user': hists_by_user, 'timestamp_split': timestamp_split,
            'seed_timestamps': seed_timestamps, 'chrono_split': chrono_split, 'target_col': target_col,
            'scaler': scaler, 'labelizer': labelizer, 'user_ids':user_ids, 'all_columns':all_columns,
            'hist_train_ranges':hist_train_ranges, 'chosen_params':chosen_params, 'model_type':model_type,
            'diss_weights':diss_weights, 'no_compat_equality_groups_per_model':no_compat_equality_groups_per_model,
            'result_type_dir':result_type_dir
        }
