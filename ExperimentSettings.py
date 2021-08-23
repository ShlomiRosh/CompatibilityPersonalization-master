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
model_names = list(models_to_test.keys())

# experiment settings -----
only_test = False
make_tradeoff_plots = True
show_tradeoff_plots = True
plot_confusion = False
verbose = False
keep_train_test_ratio = True
autotune_autc = False
keep_train_test_ratio = True
test_frac = 0.1
predetermined_timestamps = True
min_subset_size = 5

train_frac = 0.6
valid_frac = 0.3

result_dir = 'result'
# ------ experiment settings


def get_experiment_parameters(dataset_name, result_analysis=False):
    if dataset_name == 'assistment':
        # FOR MODEL TESTING
        # data settings
        target_col = 'correct'
        original_categ_cols = ['skill', 'tutor_mode', 'answer_type', 'type', 'original']
        user_cols = ['user_id']
        skip_cols = ['skill']
        df_max_size = 0
        hists_already_determined = False
        # experiment settings
        train_frac = 0.6
        valid_frac = 0.3
        h1_frac = 0.01  # if > 1 then is considered as num. of samples, not fraction
        h2_len = 10000000
        seeds = range(1)
        inner_seeds = range(1)
        weights_num = 2
        weights_range = [0, 1]
        # user settings
        min_hist_len = 50
        max_hist_len = 10000000
        min_hist_len_to_test = 0
        metrics = ['auc']
        # model settings

        model_params = {'name': 'tree',

                        'forced_params_per_model': {},

                        # 'params': {'ccp_alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]}}
                        'params': {'ccp_alpha': [0.001, 0.01]}}

        # FOR RESULT ANALYSIS
        experiment_type = 'fit on train and valid'
        version = '1 outer fold'
        user_type = 'user_id'
        performance_metric = 'auc'
        bin_size = 1

    if dataset_name == 'citizen_science':
        # FOR MODEL TESTING
        # data settings
        target_col = 'd_label'
        original_categ_cols = []
        user_cols = ['user_id']
        skip_cols = []
        # skip_cols = ['timestamp']
        # skip_cols = ['u_bHavePastSession', 'u_sessionCount', 'u_avgSessionTasks', 'u_medianSessionTasks',
        #              'u_recentAvgSessionTasks', 'u_sessionTasksvsUserMedian', 'u_sessionTasksvsRecentMedian',
        #              'u_avgSessionTime', 'u_sessionTimevsRecentAvg', 'u_sessionTimevsUserMedian',
        #              'u_sessionAvgDwellvsUserAvg', 'u_sessionAvgDwellvsRecentAvg']
        df_max_size = 0
        hists_already_determined = True
        # experiment settings
        train_frac = 0.6
        valid_frac = 0.3
        h1_frac = 0.01  # if > 1 then is considered as num. of samples, not fraction
        h2_len = 10000000
        seeds = range(5)
        inner_seeds = range(10)
        weights_num = 5
        weights_range = [0, 1]
        # user settings
        min_hist_len = 50
        max_hist_len = 10000000
        min_hist_len_to_test = 0
        metrics = ['auc']
        # model settings

        model_params = {'name': 'tree',

                        'forced_params_per_model': {},

                        'params': {'ccp_alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
                        # 'params': {'ccp_alpha': 0.001}
                        }

        # FOR RESULT ANALYSIS
        experiment_type = 'fit on train and valid'
        version = '5 outer folds'
        user_type = 'user_id'
        performance_metric = 'auc'
        bin_size = 1

    if not result_analysis:
        return [target_col, original_categ_cols, user_cols, skip_cols, hists_already_determined, df_max_size,
                train_frac, valid_frac, h1_frac, h2_len, seeds, inner_seeds, weights_num, weights_range, model_params,
                min_hist_len, max_hist_len, metrics, min_hist_len_to_test]
    else:
        return version, user_type, target_col, experiment_type, performance_metric, bin_size, min_hist_len_to_test
