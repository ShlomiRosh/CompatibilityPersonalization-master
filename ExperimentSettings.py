import numpy as np

# dataset_name = 'assistment'
# dataset_name = 'citizen_science'
dataset_name = 'mooc'

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
no_compat_equality_groups = [['no hist', 'm4', 'm6'], ['m1', 'm2', 'm3'], ['m5', 'm7', 'm8']]
no_compat_equality_groups_per_model = {'no hist': ['no hist', 'm4', 'm6'],
                                       'm4': ['no hist', 'm4', 'm6'],
                                       'm6': ['no hist', 'm4', 'm6'],
                                       'm1': ['m1', 'm2', 'm3'],
                                       'm2': ['m1', 'm2', 'm3'],
                                       'm3': ['m1', 'm2', 'm3'],
                                       'm5': ['m5', 'm7', 'm8'],
                                       'm7': ['m5', 'm7', 'm8'],
                                       'm8': ['m5', 'm7', 'm8']}

# dataset settings
dataset_dir = 'datasets/%s' % dataset_name
dataset_path = '%s/%s.csv' % (dataset_dir, dataset_name)
data_sets = {
    'assistment': {
        'models': [
            {'clf': 'tree', 'params': {'ccp_alpha': [0.001, 0.01]}},
            # {'clf': 'forest', 'params': {'ccp_alpha': [0.001, 0.01]}},
            # {'clf': 'xgboost', 'params': {'max_leaves': [0]}}
        ],
        'path': 'Datasets/assistment/assistment.csv',
        'target_col': 'correct',
        'user_cols': ['user_id'],
        'skip_cols': ['skill'],
        'original_categ_cols': ['skill', 'tutor_mode', 'answer_type', 'type', 'original'],
        'seeds': range(1),
        'inner_seeds': range(1),
        'weights_num': 2,
        'hists_already_determined': False,
        'min_hist_len': 50,
        'result_analysis': {
            'version': '1 outer fold',
            'user_type': 'user_id',
            'target_col': 'correct',
            'experiment_type': 'fit on train and valid',
            'performance_metric': 'auc',
            'bin_size': 1,
            'min_hist_len_to_test': 0
            # version:, user_type, target_col, experiment_type, performance_metric, bin_size, min_hist_len_to_test
        }
    },
    'citizen_science': {
        'models': [
            {'clf': 'tree', 'params': {'ccp_alpha': [0.0]}},
            # {'clf': 'forest', 'params': {'ccp_alpha': [0.0]}},
            # {'clf': 'xgboost', 'params': {'max_leaves': [0]}}
        ],
        'path': 'Datasets/citizen_science/citizen_science.csv',
        'target_col': 'd_label',
        'user_cols': ['user_id'],
        'skip_cols': [],
        'original_categ_cols': [],
        'seeds': range(5),
        'inner_seeds': range(10),
        'weights_num': 5,
        'hists_already_determined': True,
        'min_hist_len': 50,
        'result_analysis': {
            'version': '5 outer folds',
            'user_type': 'user_id',
            'target_col': 'd_label',
            'experiment_type': 'fit on train and valid',
            'performance_metric': 'auc',
            'bin_size': 1,
            'min_hist_len_to_test': 0
            # # version:, user_type, target_col, experiment_type, performance_metric, bin_size, min_hist_len_to_test
        },
    },
    'mooc': {
        'path': 'datasets/mooc/mooc.csv',
        'target_col': 'confusion',
        'original_categ_cols': ['course_display_name', 'post_type', 'CourseType'],
        'user_cols': ['forum_uid'],
        'skip_cols': [],
        'hists_already_determined': False,
        'seeds': range(5),
        'inner_seeds': range(10),
        'weights_num': 5,
        'weights_range': [0, 1],
        'min_hist_len': 20,

        'models': [
            {'clf': 'tree', 'params': {'ccp_alpha': [0.01, 0.1]}}
        ],
        # {'clf': 'forest', 'params': {'ccp_alpha': [0.0]}},
        # {'clf': 'xgboost', 'params': {'max_leaves': [0]}}
    },
    'result_analysis': {
        'version': '15 outer folds',
        'user_type': 'forum_uid',
        'target_col': 'confusion',
        'experiment_type': 'fit on train and valid',
        'performance_metric': 'auc',
        'bin_size': 1,
        'min_hist_len_to_test': 0
    }
}
model_params = data_sets[dataset_name]['models'][0]

# output settings
result_dir = 'result'
result_type_dir = '%s/%s' % (result_dir, data_sets[dataset_name]['user_cols'][0])
overwrite_result_folder = True
reset_cache = False

# model settings
df_max_size = 0
train_frac = 0.6
valid_frac = 0.3
test_frac = 0.1
h1_frac = 0.01  # if > 1 then is considered as num. of samples, not fraction
h2_len = 10000000
weights_range = [0, 1]
min_hist_len = 50
max_hist_len = 10000000
min_hist_len_to_test = 0
metrics = ['auc']
diss_weights = list(np.linspace(0, 1, data_sets[dataset_name]['weights_num']))

# experiment settings
autotune_autc = False
verbose = False

# NestedCrossValidation only
only_test = False
# RunnerFinal only
make_tradeoff_plots = True
# AnalyseResults and RunnerFinal
show_tradeoff_plots = True
