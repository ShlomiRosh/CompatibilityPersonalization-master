import csv
import os.path
import shutil

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

import ExperimentSettings as es


class DataPreparations:

    def __init__(self):
        self.__prepared_params = {}
        self.__run_prepartion()

    def get_experiment_parameters(self):
        return self.__prepared_params

    def __run_prepartion(self):

        self.__create_params_file()

        # run whole experiment for each user column selection
        user_col = es.data_sets[es.dataset_name]['user_cols'][0]
        data_config = es.data_sets[es.dataset_name]

        print('user column = %s' % user_col)

        # result_type_dir = '%s/%s' % (es.result_dir, user_col)
        done_by_seed = self.__write_result()

        cache_dir = '%s/caches/%s skip_%s max_len_%d min_hist_%d max_hist_%d balance_%s' % (
            es.dataset_dir, user_col, len(data_config['skip_cols']), es.df_max_size, es.min_hist_len,
            es.max_hist_len,
            False)
        if es.reset_cache and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        safe_make_dir(cache_dir)

        self.__load_seeds_in_cache(cache_dir, user_col)

        self.__prepared_params = self.__get_prepared_params(cache_dir, user_col, done_by_seed)

    def __get_prepared_params(self, cache_dir, user_col, done_by_seed):
        data_config, user_ids = es.data_sets[es.dataset_name], []
        hists_by_user, hist_train_ranges = {}, {}
        curr_h2_len, num_users_to_test = 0, 0

        print("determine experiment's users...")
        min_max_col_values = pd.read_csv('%s/all_columns.csv' % cache_dir, dtype=np.int64)
        all_columns = min_max_col_values.columns
        dataset = pd.read_csv('%s/0.csv' % cache_dir)
        groups_by_user = dataset.groupby(user_col, sort=False)

        for user_id, hist in groups_by_user:
            user_ids.append(user_id)
            hist = hist.drop(columns=[user_col])

            hist_train_len = len(hist) * es.train_frac
            if data_config['hists_already_determined'] or (
                    es.min_hist_len <= hist_train_len and curr_h2_len + hist_train_len <= es.h2_len):
                if len(hist) >= es.min_hist_len_to_test:
                    num_users_to_test += 1
                if len(hist) > es.max_hist_len:
                    hist = hist[:es.max_hist_len]
                hists_by_user[user_id] = hist
                min_max_col_values = min_max_col_values.append(hist.apply(min_and_max), sort=False)

                hist_train_ranges[user_id] = [curr_h2_len, len(hist)]
                curr_h2_len += len(hist)
        del groups_by_user

        print('cols=%d data_len=%d h1_frac=%s users=%d diss_weights=%d model_type=%s auto_tune_params=%s' % (
            len(all_columns) - 1, curr_h2_len, es.h1_frac, len(hists_by_user), len(es.diss_weights),
            es.model_params['clf'], True))

        # hists_by_user
        min_max_col_values = min_max_col_values.reset_index(drop=True)
        scaler, labelizer = MinMaxScaler(), LabelBinarizer()
        labelizer.fit(min_max_col_values[[data_config['target_col']]])
        del min_max_col_values

        prepared_params = {
            'num_users_to_test': num_users_to_test,
            'done_by_seed': done_by_seed,
            'hists_by_user': hists_by_user,
            'scaler': scaler,
            'labelizer': labelizer,
            'user_ids': user_ids,
            'all_columns': all_columns,
            'hist_train_ranges': hist_train_ranges,

            'no_compat_equality_groups_per_model': es.no_compat_equality_groups_per_model,
            'seeds': data_config['seeds'],
            'inner_seeds': data_config['inner_seeds'],
            'target_col': data_config['target_col'],
            'chosen_params': es.model_params['params'],
            'model_type': es.model_params['clf'],
            'diss_weights': es.diss_weights,
            'result_type_dir': es.result_type_dir
        }
        return prepared_params

    def __load_seeds_in_cache(self, cache_dir, user_col):
        data_config = es.data_sets[es.dataset_name]
        all_seeds_in_cache = True
        if not os.path.exists('%s/0.csv' % cache_dir):
            all_seeds_in_cache = False

        print('loading %s dataset...' % es.dataset_name)
        if all_seeds_in_cache:
            return

        dataset_full = self.__load_data(user_col)

        if data_config['hists_already_determined']:
            dataset_full.to_csv('%s/0.csv' % cache_dir, index=False)
            if not os.path.exists('%s/all_columns.csv' % cache_dir):
                pd.DataFrame(columns=list(dataset_full.drop(columns=[user_col]).columns)).to_csv(
                    '%s/all_columns.csv' % cache_dir, index=False)
        else:
            self.__sort_user_histories(dataset_full, user_col, cache_dir)
        del dataset_full

    def __load_data(self, user_col):
        data_config = es.data_sets[es.dataset_name]
        dataset_full = pd.read_csv(es.dataset_path).drop(columns=data_config['skip_cols'])
        if 'timestamp' in dataset_full.columns:
            dataset_full = dataset_full.drop(columns='timestamp')

        categ_cols = data_config['original_categ_cols']
        try:  # dont one hot encode the user_col
            categ_cols.remove(user_col)
        except ValueError:
            pass
        for category in categ_cols:
            if category in data_config['skip_cols']:
                categ_cols.remove(category)

        print('one-hot encoding the data... ')
        col_groups_dict = {}
        categs_unique_values = dataset_full[categ_cols].nunique()
        i = 0
        for col in dataset_full.columns:
            if col in [user_col, data_config['target_col']]:
                continue
            unique_count = 1
            if col in categ_cols:
                unique_count = categs_unique_values[col]
            col_groups_dict[col] = range(i, i + unique_count)
            i = i + unique_count

        dataset_full = ce.OneHotEncoder(cols=categ_cols, use_cat_names=True).fit_transform(dataset_full)

        return dataset_full

    def __sort_user_histories(self, dataset_full, user_col, cache_dir):
        data_config = es.data_sets[es.dataset_name]

        print('sorting histories...')
        groups_by_user = dataset_full.groupby(user_col, sort=False)
        dataset_full = dataset_full.drop(columns=[user_col])
        all_columns = list(dataset_full.columns)
        if not os.path.exists('%s/all_columns.csv' % cache_dir):
            pd.DataFrame(columns=all_columns).to_csv('%s/all_columns.csv' % cache_dir, index=False)

        # get user histories
        for seed in data_config['seeds']:
            if not os.path.exists('%s/%d.csv' % (cache_dir, seed)):
                hists = {}
                for user_id in groups_by_user.groups.keys():
                    hist = groups_by_user.get_group(user_id).drop(columns=[user_col])
                    if len(hist) < es.min_hist_len:
                        continue

                    hists[user_id] = hist
                sorted_hists = [[k, v] for k, v in reversed(sorted(hists.items(), key=lambda n: len(n[1])))]
                seed_df = pd.DataFrame(columns=[user_col] + all_columns, dtype=np.int64)
                for user_id, hist in sorted_hists:
                    hist[user_col] = [user_id] * len(hist)
                    seed_df = seed_df.append(hist, sort=False)
                seed_df.to_csv('%s/0.csv' % cache_dir, index=False)
            # if not balance_histories:
            break
        del groups_by_user
        del hists

    def __write_result(self):

        done_by_seed = {}
        header = get_results_header()

        # create all folders
        if not os.path.exists(es.result_type_dir):
            for metric in es.metrics:
                os.makedirs('%s/%s' % (es.result_type_dir, metric))
                for subset in ['train', 'valid', 'test']:
                    with open('%s/%s/%s_log.csv' % (es.result_type_dir, metric, subset), 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(header)
        else:  # load what is already done
            df_done = pd.read_csv('%s/%s/test_log.csv' % (es.result_type_dir, es.metrics[-1]))
            groups_by_seed = df_done.groupby('seed')
            for seed, seed_group in groups_by_seed:
                done_by_inner_seed = {}
                done_by_seed[seed] = done_by_inner_seed
                groups_by_inner_seed = seed_group.groupby('inner_seed')
                for inner_seed, inner_seed_group in groups_by_inner_seed:
                    done_by_inner_seed[inner_seed] = len(pd.unique(inner_seed_group['user']))
            del df_done

        return done_by_seed

    def __create_params_file(self):
        data_config = es.data_sets[es.dataset_name]

        if es.overwrite_result_folder and os.path.exists(es.result_dir):
            shutil.rmtree(es.result_dir)
        if not os.path.exists(es.result_dir):
            os.makedirs(es.result_dir)
            with open('%s/parameters.csv' % es.result_dir, 'w', newline='') as file_out:
                writer = csv.writer(file_out)
                writer.writerow(['train_frac', 'valid_frac', 'dataset_max_size', 'h1_frac', 'h2_len', 'seeds',
                                 'inner_seeds', 'weights_num', 'weights_range', 'min_hist_len', 'max_hist_len',
                                 'skip_cols', 'model_type', 'params'])
                writer.writerow(
                    [es.train_frac, es.valid_frac, es.df_max_size, es.h1_frac, es.h2_len, len(data_config['seeds']),
                     len(data_config['inner_seeds']),
                     data_config['weights_num'], str(es.weights_range), es.min_hist_len, es.max_hist_len,
                     str(data_config['skip_cols']), es.model_params['clf'], es.model_params['params']])


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def min_and_max(x):
    return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])


def get_results_header():
    header = ['user', 'len', 'seed', 'inner_seed', 'h1_acc', 'weight']
    for model_name in es.model_names:
        header.extend(['%s x' % model_name, '%s y' % model_name])
    return header
