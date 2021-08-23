import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import auc

from ExperimentSettings import get_experiment_parameters


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_users(log_dir, log_set):
    df_log = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    log_by_users = df_log.groupby('user')
    print('\nsplitting %s log into user logs...' % log_set)
    user_idx = 0
    for user_id, user_data in log_by_users:
        user_idx += 1
        print('\tuser %d/%d' % (user_idx, len(log_by_users)))
        try:
            mod_user_id = []
            for row in user_data['seed']:
                mod_user_id += [str(user_id) + '_' + str(row)]
            user_data['user'] = mod_user_id
        except KeyError:
            user_data['user'] = user_id
        user_data.to_csv('%s/users_%s/logs/%s_%s.csv' % (log_dir, log_set, log_set, user_id), index=False)


def get_model_dict(cmap_name):
    models = {
        'no hist': {'sample_weight': [1, 1, 0, 0], 'color': 'black', 'std': True},
        'best_valid': {'sample_weight': ['', '', '', ''], 'color': 'red', 'std': True},
        'best_test': {'sample_weight': ['', '', '', ''], 'color': 'gold', 'std': True},
    }
    if add_parametrized_models:
        parametrized_models = [  # [general_loss, general_diss, hist_loss, hist_diss]
            ['m1', [0, 0, 1, 1], False],
            ['m2', [0, 1, 1, 0], False],
            ['m3', [0, 1, 1, 1], False],
            ['m4', [1, 0, 0, 1], False],
            ['m5', [1, 0, 1, 1], False],
            ['m6', [1, 1, 0, 1], False],
            ['m7', [1, 1, 1, 0], False],
            ['m8', [1, 1, 1, 1], False],
        ]
        cmap = plt.cm.get_cmap(cmap_name)
        for i in range(len(parametrized_models)):
            model = parametrized_models[i]
            models[model[0]] = {'sample_weight': model[1], 'color': cmap((i + 1) / (len(parametrized_models) + 3)),
                                'std': model[2]}
    return models


def plot_results(log_dir, dataset, models, log_set, compare_by_percentage, bin_size=1, user_name='',
                 show_tradeoff_plots=False, smooth_color_progression=False, std_opacity=0.15,
                 performance_metric='AUC', prefix='', diss_labels=False,
                 plot_markers=True, weight_by_len=True):
    if user_name == '':
        df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    else:
        df_results = pd.read_csv('%s/logs/%s_%s.csv' % (log_dir, log_set, user_name))

    model_names = [i[:-2] for i in df_results.columns if ' x' in i and i[:-2] in models.keys()]
    xs, ys, xs_plot, ys_plot = [], [], [], []
    stds = {}
    autcs_average = []
    if weight_by_len:
        h1_avg_acc = np.average(df_results['h1_acc'], weights=df_results['len'])
    else:
        h1_avg_acc = np.average(df_results['h1_acc'])

    weights = pd.unique(df_results['weight'])
    groups_by_weight = df_results.groupby('weight')

    marker = None
    if plot_markers:
        marker = 's'

    cmap = plt.cm.get_cmap('jet')

    df_by_weight = [groups_by_weight.get_group(i) for i in weights]
    df_by_weight_norm = None
    for model_name in model_names:
        if weight_by_len:
            x = [np.average(i['%s x' % model_name], weights=i['len']) for i in df_by_weight]
            y = [np.average(i['%s y' % model_name], weights=i['len']) for i in df_by_weight]
        else:
            x = [np.average(i['%s x' % model_name]) for i in df_by_weight]
            y = [np.average(i['%s y' % model_name]) for i in df_by_weight]

        plot_std = models[model_name]['std']
        if plot_std:
            if df_by_weight_norm is None:
                model_names_for_std = [i for i in model_names if models[i]['std']]
                df_by_weight_norm = get_df_by_weight_norm(df_results, weights, model_names_for_std)
            std = [df_by_weight_norm[i]['%s y' % model_name].std() for i in range(len(weights))]
        if compare_by_percentage:
            h1_area = (x[-1] - x[0]) * h1_avg_acc
            autc = auc(x, y) - h1_area
        else:
            autc = auc(x, y)
        autcs_average.append(autc)
        if model_name == 'no hist':
            no_hist_autc = autc

        xs.append(x)
        ys.append(y)
        if bin_size > 1:  # bin points
            last_idx = len(x) - ((len(x) - 2) % bin_size) - 1
            x_binned = np.mean(np.array(x[1:last_idx]).reshape(-1, bin_size), axis=1)
            y_binned = np.mean(np.array(y[1:last_idx]).reshape(-1, bin_size), axis=1)
            xs_plot.append([x[0]] + list(x_binned) + [np.mean(x[last_idx:-1]), x[-1]])
            ys_plot.append([y[0]] + list(y_binned) + [np.mean(y[last_idx:-1]), y[-1]])
            if plot_std:
                std_binned = np.mean(np.array(std[1:last_idx]).reshape(-1, bin_size), axis=1)
                stds[model_name] = [std[0]] + list(std_binned) + [np.mean(std[last_idx:-1]), std[-1]]
        else:
            xs_plot.append(x)
            ys_plot.append(y)
            if plot_std:
                stds[model_name] = std

    min_x = xs[0][0]
    max_x = xs[0][-1]
    h1_x = [min_x, max_x]
    h1_y = [h1_avg_acc, h1_avg_acc]

    autc_improvs = []
    for i in range(len(model_names)):
        autc = autcs_average[i]
        if compare_by_percentage:
            autc_improvs.append((autc / no_hist_autc - 1) * 100)
        else:
            autc_improvs.append(autc - no_hist_autc)

    sorted_idxs = [idx for autc, idx in reversed(sorted(zip(autc_improvs, range(len(autc_improvs)))))]

    cell_text = []
    model_names_sorted = []
    colors = []

    if bin_size > 1:  # bin points
        last_idx = len(weights) - ((len(weights) - 2) % bin_size) - 1
        weights_binned = np.mean(np.array(weights[1:last_idx]).reshape(-1, bin_size), axis=1)
        weights = [weights[0]] + list(weights_binned) + [np.mean(weights[last_idx:-1]), weights[-1]]

    color_idx = 1
    for i in sorted_idxs:
        x_plot, y_plot = xs_plot[i], ys_plot[i]
        autc_improv = autc_improvs[i]
        if autc_improv >= 0:
            sign = '+'
        else:
            sign = ''
        model_name = model_names[i]
        model_names_sorted.append(model_name)
        model = models[model_name]
        if smooth_color_progression:
            if model_name == 'no hist':
                color = 'black'
            else:
                color = cmap(color_idx / (len(model_names) + 1))
                color_idx += 1
        else:
            color = model['color']

        sample_weight = model['sample_weight']
        if model_name == 'sim_ann':
            sample_weight = ['%.3f' % i for i in sample_weight]
        cell_text += [sample_weight + ['%s%.1f%%' % (sign, autc_improv)]]
        if compare_by_percentage:
            label = '%s (%s%.1f%%)' % (model_name, sign, autc_improv)
        else:
            label = '%s (%.5f)' % (model_name, autc_improv)
        if model_name == 'no hist':
            plt.plot(x_plot, y_plot, label='baseline', color=color, marker=marker)
        else:
            plt.plot(x_plot, y_plot, label=label, color=color, marker=marker)
        if diss_labels:
            for i in range(len(x_plot)):
                x_i, y_i = x_plot[i], y_plot[i]
                plt.text(x_i, y_i + 0.005, '%.2f' % weights[i], color=color, ha='center')
        if model['std']:
            y = np.array(y_plot)
            s = np.array(stds[model_name])
            plt.fill_between(x_plot, y + s, y - s, facecolor=color, alpha=std_opacity)
        if model_name == 'no hist':
            color = 'white'
        colors.append(color)
    plt.plot(h1_x, h1_y, 'k--', marker='.', label='pre-update model')

    if user_name == '':
        title = '%s dataset%s' % (dataset, prefix)
        save_name = '%s/%s_plots%s.png' % (log_dir, log_set, prefix)
    else:
        len_h = df_results.loc[0]['len']
        title = 'dataset=%s user=%s len(h)=%d' % (dataset, user_name, len_h)
        save_name = '%s/%s_%s.png' % (log_dir, log_set, user_name)
    plt.xlabel('compatibility')
    plt.ylabel(performance_metric)
    plt.title(title)
    plt.legend()
    plt.savefig(save_name, bbox_inches='tight')
    if show_tradeoff_plots:
        plt.show()
    plt.clf()


def get_df_by_weight_norm(df, weights, model_names):
    drop_columns = [i for i in df.columns if (' x' in i or (' y' in i and i[:-2] not in model_names))]
    drop_columns += ['user', 'inner_seed']
    df = df.drop(columns=drop_columns)
    df_dict = {'weight': []}
    for model_name in model_names:
        df_dict['%s y' % model_name] = []
    seed_groups = df.groupby('seed')
    for seed, seed_group in seed_groups:
        df_dict['weight'].extend(weights)
        weight_groups = seed_group.groupby('weight')
        models_y = {i: [] for i in ['h1_acc'] + model_names}
        for weight, weight_group in weight_groups:
            for model_name in ['h1_acc'] + model_names:
                col_name = model_name
                if model_name != 'h1_acc':
                    col_name += ' y'
                models_y[model_name].append(np.average(weight_group[col_name], weights=weight_group['len']))
        for i, j in models_y.items():
            models_y[i] = np.array(j)
        for model_name in model_names:
            model_y = models_y[model_name]

            if model_name == 'no hist':
                no_hist_y = model_y
            model_y_norm = model_y - no_hist_y

            df_dict['%s y' % model_name].extend(list(model_y_norm))

    df_norm = pd.DataFrame(df_dict)
    groups_by_weight = df_norm.groupby('weight')
    return [groups_by_weight.get_group(i) for i in weights]


def get_best_models(log_dir, models, log_set, user_name='', plot_tradeoffs=False):
    if user_name == '':
        df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    else:
        df_results = pd.read_csv('%s/%s_%s.csv' % (log_dir, log_set, user_name))

    if plot_tradeoffs:
        seed_plots_dir = '%s/%s_seed_plots' % (log_dir, log_set)
        safe_make_dir(seed_plots_dir)

    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    seeds = pd.unique(df_results['seed']).tolist()
    groups_by_seed = df_results.groupby('seed')
    weights = pd.unique(df_results['weight'])
    best_models_by_seed = []

    for seed_idx in range(len(seeds)):
        seed = seeds[seed_idx]
        print('\t%d/%d seed %d' % (seed_idx + 1, len(seeds), seed))
        df_seed = groups_by_seed.get_group(seed)
        groups_by_weight = df_seed.groupby('weight')
        if user_name == '':
            h1_avg_acc = np.average(df_seed['h1_acc'], weights=df_seed['len'])
            dfs_by_weight = [groups_by_weight.get_group(i) for i in weights]
        else:
            h1_avg_acc = np.mean(df_seed['h1_acc'])
            means = groups_by_weight.mean()

        autcs = []
        xs_seed, ys_seed = [], []
        for model_name in model_names:
            if model_name not in models.keys():
                continue
            if user_name == '':
                x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
                y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]
            else:
                x = means['%s x' % model_name].tolist()
                y = means['%s y' % model_name].tolist()

            h1_area = (x[-1] - x[0]) * h1_avg_acc
            autc = auc(x, y) - h1_area
            autcs.append(autc)
            xs_seed.append(x)
            ys_seed.append(y)
        min_x = min(min(i) for i in xs_seed)
        max_x = max(max(i) for i in xs_seed)

        # get best model by seed
        best_model = ''
        best_autc = None
        for i in range(len(model_names)):
            model_name = model_names[i]
            color = models[model_name]['color']
            if model_name not in models.keys():
                continue
            autc = autcs[i]
            if best_autc is None or autc > best_autc:
                best_autc = autc
                best_model = model_name
            if plot_tradeoffs:
                plt.plot(xs_seed[i], ys_seed[i], label='%s autc=%.5f' % (model_name, autc), color=color)
        if plot_tradeoffs:
            plt.plot([min_x, max_x], [h1_avg_acc, h1_avg_acc], 'k--', label='h1', marker='.')
            plt.xlabel('compatibility')
            plt.ylabel('accuracy')
            plt.legend()
            plt.title('user=%s seed=%d best=%s' % (user_name, seed, best_model))
            plt.savefig('%s/user_%s seed_%d' % (seed_plots_dir, user_name, seed), bbox_inches='tight')
            plt.clf()
        best_models_by_seed.append(best_model)
    return seeds, best_models_by_seed
    # todo: return best model by weight


def add_best_model(log_dir, from_set, to_set):
    df_best = pd.read_csv('%s/best_models_%s.csv' % (log_dir, from_set))
    df_to = pd.read_csv('%s/%s_log.csv' % (log_dir, to_set))
    groups_by_user = df_to.groupby('user')
    user_names = pd.unique(df_best['user'])
    seeds = pd.unique(df_best['seed'])
    best_models = {user: [list(row) for i, row in data.iterrows()] for user, data in df_best.groupby('user')}
    new_model_x = []
    new_model_y = []
    for user_idx, user_name in enumerate(user_names):
        print('\tuser %d/%d' % (user_idx + 1, len(user_names)))
        df_user = groups_by_user.get_group(user_name)
        groups_by_seed = df_user.groupby('seed')
        for seed_idx, seed in enumerate(seeds):
            df_seed = groups_by_seed.get_group(seed)
            best_user, best_seed, best_model = best_models[user_name][seed_idx]
            new_model_x.extend(df_seed['%s x' % best_model].tolist())
            new_model_y.extend(df_seed['%s y' % best_model].tolist())
            if user_name != best_user or seed != best_seed:
                raise ValueError('results and best lists not in same order of user -> seed')

    from_set_name = from_set.split('_')[0]
    df_to['best_%s x' % from_set_name] = new_model_x
    df_to['best_%s y' % from_set_name] = new_model_y
    df_to.to_csv('%s/%s_with_best_log.csv' % (log_dir, to_set), index=False)


def binarize_results_by_compat_values(log_dir, log_set, num_bins=100, print_progress=True):
    bins = np.array([i / num_bins for i in range(num_bins + 1)])
    # bins = np.linspace(0, 1, num_bins)
    df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    dict_binarized = {i: [] for i in df_results.columns}
    user_names = pd.unique(df_results['user'])
    groups_by_user = df_results.groupby('user')
    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    seeds = None
    inner_seeds = None
    missing_values = []

    for user_idx, user_name in enumerate(user_names):
        if print_progress:
            print('%d/%d user=%s' % (user_idx + 1, len(user_names), user_name))
        df_user = groups_by_user.get_group(user_name)
        user_len = df_user['len'].max()
        user_name_repeated = [user_name] * (num_bins + 1)
        user_len_repeated = [user_len] * (num_bins + 1)
        if seeds is None:
            seeds = pd.unique(df_user['seed'])
        groups_by_seed = df_user.groupby('seed')

        for seed_idx, seed in enumerate(seeds):
            if print_progress:
                print('\t%d/%d seed=%d' % (seed_idx + 1, len(seeds), seed))
            seed_repeated = [seed] * (num_bins + 1)
            try:
                df_seed = groups_by_seed.get_group(seed)
            except KeyError:
                missing_values.append('user=%s seed=%d' % (user_name, seed))
                continue
            if inner_seeds is None:
                inner_seeds = pd.unique(df_seed['inner_seed'])
            groups_by_inner_seed = df_seed.groupby('inner_seed')

            for inner_seed_idx, inner_seed in enumerate(inner_seeds):
                try:
                    df_inner_seed = groups_by_inner_seed.get_group(inner_seed)
                except KeyError:
                    missing_values.append('user=%s seed=%d inner_seed=%d' % (user_name, seed, inner_seed))
                    continue
                if len(missing_values) != 0:
                    continue
                h1_acc = df_inner_seed['h1_acc'].iloc[0]
                inner_seed_repeated = [inner_seed] * (num_bins + 1)
                h1_acc_repeated = [h1_acc] * (num_bins + 1)

                dict_binarized['user'].extend(user_name_repeated)
                dict_binarized['len'].extend(user_len_repeated)
                dict_binarized['seed'].extend(seed_repeated)
                dict_binarized['inner_seed'].extend(inner_seed_repeated)
                dict_binarized['h1_acc'].extend(h1_acc_repeated)
                dict_binarized['weight'].extend(bins)
                xs = []
                ys = []
                for model_name in model_names:
                    x = df_inner_seed['%s x' % model_name].tolist()
                    y = df_inner_seed['%s y' % model_name].tolist()
                    for i in range(1, len(x)):  # make x monotonically increasing
                        if x[i] < x[i - 1]:
                            x[i] = x[i - 1]
                    xs.append(x)
                    ys.append(y)

                # add min and max x to each model
                min_x = min([min(i) for i in xs])
                max_x = max([max(i) for i in xs])
                x_bins = (bins * (max_x - min_x) + min_x).tolist()
                # to avoid floating point weirdness in first and last values
                x_bins[0] = min_x
                x_bins[-1] = max_x
                for i in range(len(model_names)):
                    x = xs[i]
                    y = ys[i]
                    xs[i] = [min_x] + x + [x[-1], max_x]
                    ys[i] = [y[0]] + y + [h1_acc, h1_acc]

                # binarize
                for model_idx in range(len(model_names)):
                    model_name = model_names[model_idx]
                    x = xs[model_idx]
                    y = ys[model_idx]
                    y_bins = []
                    j = 0
                    for x_bin in x_bins:  # get y given x for each x_bin
                        while not x[j] <= x_bin <= x[j + 1]:
                            j += 1
                        x_left, x_right, y_left, y_right = x[j], x[j + 1], y[j], y[j + 1]
                        if x_left == x_right:  # vertical line
                            y_bin = max(y_left, y_right)
                        else:
                            slope = (y_right - y_left) / (x_right - x_left)
                            y_bin = y_left + slope * (x_bin - x_left)
                        y_bins.append(y_bin)
                    dict_binarized['%s x' % model_name].extend(x_bins)
                    dict_binarized['%s y' % model_name].extend(y_bins)
    if len(missing_values) != 0:
        with open('%s/%s_missing_values.txt' % (log_dir, log_set), 'w') as file:
            for missing_value in missing_values:
                file.write('%s\n' % missing_value)
        print(missing_values)
        raise KeyError('missing values!')
    pd.DataFrame(dict_binarized).to_csv('%s/%s_bins_log.csv' % (log_dir, log_set), index=False)


def best_count_values(log_dir, log_set):
    df_best = pd.read_csv('%s/best_models_%s.csv' % (log_dir, log_set))
    groups_by_users = df_best.groupby('user')
    models = pd.unique(df_best['model'])
    users = pd.unique(df_best['user'])
    dict_counts = {i: [] for i in models}
    dict_counts['user'] = users
    for user in users:
        df_user = groups_by_users.get_group(user)
        value_counts = df_user['model'].value_counts()
        for model in models:
            try:
                dict_counts[model].append(0)
            except AttributeError:
                print('hi')
        for model in list(value_counts.index):
            dict_counts[model][-1] = dict_counts[model][-1] + value_counts[model]
    df_result = pd.DataFrame(dict_counts)
    df_result.to_csv('%s/best_models_%s_counts.csv' % (log_dir, log_set), index=False)


def get_autcs_averaged_over_inner_seeds(log_dir, log_set):
    df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    users = pd.unique(df_results['user'])
    user_groups = df_results.groupby('user')
    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    with open('%s/%s_autcs.csv' % (log_dir, log_set), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user', 'len', 'seed', 'h1'] + [i for i in model_names])
        for user_idx, user in enumerate(users):
            print('\tuser %d/%d' % (user_idx + 1, len(users)))
            df_user = user_groups.get_group(user)
            hist_len = df_user.iloc[0]['len']
            for seed, df_seed in df_user.groupby('seed'):
                row = [user, hist_len, seed]
                if make_geometric_average:
                    means = df_seed.groupby('weight').mean()
                    h1_y = means['h1_acc'].iloc[0]
                    for i, model_name in enumerate(model_names):
                        x = means['%s x' % model_name].tolist()
                        y = means['%s y' % model_name].tolist()
                        if i == 0:
                            row.append(h1_y * (x[-1] - x[0]))
                        autc = auc(x, y)
                        row.append(autc)
                else:
                    autc_by_model = [[] for i in model_names]
                    h1_ys = []
                    for inner_seed, df_inner_seed in df_seed.groupby('inner_seed'):
                        h1_y = df_inner_seed['h1_acc'].iloc[0]
                        for i, model_name in enumerate(model_names):
                            x = df_inner_seed['%s x' % model_name].tolist()
                            y = df_inner_seed['%s y' % model_name].tolist()
                            if i == 0:
                                h1_ys.append(h1_y * (x[-1] - x[0]))
                            autc = auc(x, y)
                            # if remove_h1_area:
                            #     autc -= h1_y * (x[-1] - x[0])
                            autc_by_model[i].append(autc)
                    row.append(np.mean(h1_ys))
                    row.extend([np.mean(i) for i in autc_by_model])
                writer.writerow(row)


def get_all_autcs(log_dir, log_set):
    df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    users = pd.unique(df_results['user'])
    user_groups = df_results.groupby('user')
    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    with open('%s/%s_all_autcs.csv' % (log_dir, log_set), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user', 'seed', 'inner seed'] + [i for i in model_names])
        for user_idx, user in enumerate(users):
            print('\tuser %d/%d' % (user_idx + 1, len(users)))
            df_user = user_groups.get_group(user)
            for seed, df_seed in df_user.groupby('seed'):
                for inner_seed, df_inner_seed in df_seed.groupby('inner_seed'):
                    row = [user, seed, inner_seed]
                    for model_name in model_names:
                        x = df_inner_seed['%s x' % model_name].tolist()
                        y = df_inner_seed['%s y' % model_name].tolist()
                        row += [auc(x, y)]
                    writer.writerow(row)


def get_best_from_row(rows, models):
    bests = []
    for row in rows:
        best_autc = 0
        best_name = 'no hist'
        best_idx = 0
        for i, model in enumerate(models):
            if row[model] > best_autc:
                best_autc = row[model]
                best_name = model
                best_idx = i
        best_vector = [0] * len(models)
        best_vector[best_idx] = 1
        bests.append([best_name, best_vector])
    return bests


def get_sample_indexes_from_user_indexes(fold, num_seeds):
    train_index, test_index = [], []
    for users_index, subset_index in zip(fold, [train_index, test_index]):
        for i in users_index:
            start_index = i * num_seeds
            subset_index.extend([start_index + j for j in range(num_seeds)])
    return np.array(train_index), np.array(test_index)


def make_one_hot(labels, models):
    rows = []
    for label in labels:
        row = [0] * len(models)
        for i, model in enumerate(models):
            if label == model:
                row[i] = 1
                break
        rows.append(row)
    return np.array(rows)


def count_labels(versions):
    global_counts = {'version': [], 'baseline': [], 'train': [], 'valid': [], 'baseline+train': [],
                     'baseline+valid': [], 'valid+train': [], 'baseline+train+valid': []}
    for version in versions:
        df = pd.read_csv('meta_dataset_ver_%s.csv' % version)
        counts = {'baseline': 0, 'train': 0, 'valid': 0, 'baseline+train': 0, 'baseline+valid': 0, 'valid+train': 0,
                  'baseline+train+valid': 0}
        for i, row in df.iterrows():
            label = row['label']
            if label == 'baseline':
                counts['baseline'] += 1
            elif label == 'train':
                counts['train'] += 1
                if row['score(baseline, train)'] == 1.0:
                    counts['baseline'] += 1
                    counts['baseline+train'] += 1
            elif label == 'valid':
                counts['valid'] += 1
                baseline = False
                if row['score(baseline, valid)'] == 1.0:
                    baseline = True
                    counts['baseline'] += 1
                    counts['baseline+valid'] += 1
                if row['score(best_train, valid)'] == 1.0:
                    counts['train'] += 1
                    counts['valid+train'] += 1
                    if baseline:
                        counts['baseline+train+valid'] += 1
        global_counts['version'].append(version)
        for key, val in counts.items():
            global_counts[key].append(val)
    pd.DataFrame(global_counts).to_csv('meta_datasets_label_counts.csv', index=False)


def compare_models(log_dir, log_set):
    df = pd.read_csv('%s/%s_autcs.csv' % (log_dir, log_set))
    models = list(df.columns)[4:]

    with open('%s/model_comparison.csv' % log_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'AUTC mean', 'AUTC std', 'vs base %', 'vs base p-val',
                         '%times > base', '%times >= base'])

        if remove_h1_area:
            for model in models:
                df[model] = df[model] - df['h1']
        groups_by_seed = df.groupby('seed')
        rows = []
        cols = list(df.columns[4:])
        for seed, group in groups_by_seed:
            row = [seed]
            for col in cols:
                if compare_models_by_weighted_avg:
                    row.append(np.average(group[col], weights=group['len']))
                else:
                    row.append(np.average(group[col]))
            rows.append(row)
        df_by_seed = pd.DataFrame(rows, columns=[seed] + cols)
        means = df_by_seed.mean()
        stds = df_by_seed.std()
        writer.writerow(['no hist', means['no hist'], stds['no hist'], '', '', '', ''])
        print()
        for model in models[1:]:  # excluding baseline
            improvement = (means[model] / means['no hist'] - 1) * 100
            t_stat, p_val = ttest_rel(df['no hist'], df[model])
            success_str = ''
            if improvement > 0 and p_val <= 0.05:
                success_str = ' SUCCESS!'
            perc_better = (df[model] > df['no hist']).mean()
            perc_eq_better = (df[model] >= df['no hist']).mean()
            line = 'model = %s improvement = %.2f%% p_val = %.5f %%>base = %s%s' % (
                model, improvement, p_val / 2, perc_better, success_str)
            print(line)
            writer.writerow([model, means[model], stds[model], improvement, p_val / 2, perc_better, perc_eq_better])
        t_stat, p_val = ttest_rel(df['best_valid'], df['best_test'])
        line = 'best_valid vs best_test: p_val = %.5f' % (p_val / 2)
        print(line)


def execute_phase(phase, log_set):
    binarize_by_compat = False
    individual_users = False
    get_best = False
    add_best = False
    get_autcs = False
    comparing_models = False
    count_best = False
    test_set = 'test_bins'

    print('\n%s' % phase)
    if phase == 'binarize validation results':
        log_set = 'valid'
        binarize_by_compat = True
    elif phase == 'binarize train results':
        log_set = 'train'
        binarize_by_compat = True
    elif phase == 'get best_u for each user using binarized validation results':
        log_set = 'valid_bins'
        individual_users = True
        get_best = True
    elif phase == 'get best_u for each user using binarized train results':
        log_set = 'train_bins'
        individual_users = True
        get_best = True
    elif phase == 'get best_u for each user using binarized test results':
        log_set = 'test_bins'
        individual_users = True
        get_best = True
    elif phase == 'binarize test results':
        log_set = 'test'
        binarize_by_compat = True
    elif phase == 'add best_u computed from validation to binarized test results':
        log_set = 'valid_bins'
        add_best = True
    elif phase == 'add best_u computed from train to binarized validation results':
        log_set = 'train_bins'
        test_set = 'valid_bins'
        add_best = True
    elif phase == 'add best_u computed from test to binarized test with best results':
        log_set = 'test_bins'
        test_set = 'test_bins_with_best'
        add_best = True
    elif phase == 'generate averaged plots for binarized test results with best':
        log_set = 'test_bins_with_best'
    elif phase == 'generate averaged plots for binarized test results with both best':
        log_set = 'test_bins_with_best_with_best'
    elif phase == 'generate individual user plots for test bins with best results':
        log_set = 'test_bins_with_best_with_best'
        individual_users = True
    elif phase == 'generate user plots for binarized train results':
        log_set = 'train_bins'
        individual_users = True
    elif phase == 'generate user plots for binarized validation results':
        log_set = 'valid_bins'
        individual_users = True
    elif phase == 'generate user plots for binarized test results':
        log_set = 'test_bins'
        individual_users = True
    elif phase == 'generate averaged plots for binarized validation results':
        log_set = 'valid_bins'
    elif phase == 'generate averaged plots for binarized test results':
        log_set = 'test_bins'
    elif phase == 'binarize train results':
        log_set = 'train'
        binarize_by_compat = True
    elif phase == 'generate averaged plots for binarized train results':
        log_set = 'train_bins'
    elif phase == 'get autcs averaged over inner seeds for train bins':
        log_set = 'train_bins'
        get_autcs = True
    elif phase == 'get autcs averaged over inner seeds for validation bins':
        log_set = 'valid_bins'
        get_autcs = True
    elif phase == 'get autcs averaged over inner seeds for test bins':
        log_set = 'test_bins'
        get_autcs = True
    elif phase == 'get autcs averaged over inner seeds for test bins with best':
        log_set = 'test_bins_with_best_with_best'
        get_autcs = True
    elif phase == 'compare models for test bins with best':
        log_set = 'test_bins_with_best_with_best'
        comparing_models = True
    elif phase == 'generate averaged plots for binarized validation results with best':
        log_set = 'valid_bins_with_best'
    elif phase == 'get best for each user':
        individual_users = True

    results_dir = 'result'
    log_dir = '%s/%s/%s' % (results_dir, user_type, performance_metric)
    models = get_model_dict('jet')

    if add_best:
        add_best_model(log_dir, log_set, test_set)
    elif count_best:
        best_count_values(log_dir, log_set)
    elif binarize_by_compat:
        binarize_results_by_compat_values(log_dir, log_set, num_normalization_bins)
    elif get_autcs:
        get_autcs_averaged_over_inner_seeds(log_dir, log_set)
    elif comparing_models:
        compare_models(log_dir, log_set)
    elif not individual_users:  # make sure this is last elif
        if get_best:
            print('got best models for general set, not individual users!')
        else:
            plot_results(log_dir, dataset, models, log_set, compare_by_percentage,
                         bin_size=bin_size, show_tradeoff_plots=True, weight_by_len=compare_models_by_weighted_avg)
    else:
        users_dir = '%s/users_%s' % (log_dir, log_set)
        user_logs_dir = '%s/logs' % users_dir
        if not os.path.exists(users_dir):
            safe_make_dir(user_logs_dir)
            split_users(log_dir, log_set)
        df = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
        user_ids = pd.unique(df['user'])
        user_groups = df.groupby('user')
        if get_best:
            user_col = []
            seed_col = []
            model_col = []
            for user_idx in range(len(user_ids)):
                user_id = user_ids[user_idx]
                print('%d/%d user %s' % (user_idx + 1, len(user_ids), user_id))
                seeds, best_models_by_seed = get_best_models('%s/users_%s/logs' % (log_dir, log_set), models,
                                                             log_set,
                                                             user_name=user_id)
                user_col += [user_id] * len(seeds)
                seed_col += seeds
                model_col += best_models_by_seed
            df = pd.DataFrame({'user': user_col, 'seed': seed_col, 'model': model_col})
            df.to_csv('%s/best_models_%s.csv' % (log_dir, log_set), index=False)
        else:
            for user_idx in range(len(user_ids)):
                user_id = user_ids[user_idx]
                print('%d/%d user=%s' % (user_idx + 1, len(user_ids), user_id))
                plot_results('%s/users_%s' % (log_dir, log_set), dataset, models, log_set,
                             compare_by_percentage, bin_size=bin_size, user_name=user_id)


add_parametrized_models = True
num_normalization_bins = 10

if __name__ == "__main__":

    phases = [
        # MAIN ANALYSIS:
        'binarize validation results',
        'binarize test results',
        'get best_u for each user using binarized validation results',
        'add best_u computed from validation to binarized test results',
        'get best_u for each user using binarized test results',
        'add best_u computed from test to binarized test with best results',
        'generate averaged plots for binarized test results with both best',
        'get autcs averaged over inner seeds for test bins with best',
        'compare models for test bins with best',
        #
        # EXTRAS:
        # 'generate averaged plots for binarized test results with best',  # phase 5
        # 'generate individual user plots for test bins with best results',  # phase 6
        # 'binarize train results',
        # 'generate averaged plots for binarized train results',
        # 'generate averaged plots for binarized validation results',
        # 'generate averaged plots for binarized test results',
        # 'generate user plots for binarized train results',
        # 'generate user plots for binarized validation results',
        # 'generate user plots for binarized test results',
        # 'get autcs averaged over inner seeds',
    ]

    from_current_result = True  # leave this true

    dataset = 'assistment'
    # dataset = 'citizen_science'
    # dataset = 'mooc'

    compare_by_percentage = False
    compare_models_by_weighted_avg = True
    remove_h1_area = True
    make_geometric_average = False

    statistic_test = ttest_rel
    # statistic_test = mannwhitneyu

    summary_metrics = ['avg']

    log_set = 'test_bins_with_best'

    params = get_experiment_parameters(dataset, True)
    version, user_type, target_col, model_type, performance_metric, bin_size, min_hist_len_to_test = params
    print('dataset = %s' % dataset)
    for phase in phases:
        execute_phase(phase, log_set)
    print('\ndone')
