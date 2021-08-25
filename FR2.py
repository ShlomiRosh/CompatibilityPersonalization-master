import csv
from time import time

import numpy as np
import pandas as pd

import AnalyseResults
import DP5 as dp
import ES5 as es
import NCV2 as ncv


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


def set_timers(params, timers_params):
    timers_params['timer_evaluating_params'] = ModuleTimer(
        len(params['seeds']) * len(params['inner_seeds']) * params['num_users_to_test'])
    timers_params['timer_validation_results'] = ModuleTimer(
        len(params['seeds']) * len(params['inner_seeds']) * params['num_users_to_test'])
    timers_params['timer_test_results'] = ModuleTimer(len(params['seeds']) * params['num_users_to_test'])
    timers_params['timers'] = [timers_params['timer_evaluating_params'], timers_params['timer_validation_results'],
                               timers_params['timer_test_results']]
    timers_params['iterations'] = sum([timer.iterations for timer in timers_params['timers']])


def write_rows_to_all_logs(params, loops_params):
    # write rows to all logs in one go to avoid discrepancies between logs
    for metric_idx, metric in enumerate(es.metrics):
        for subset_idx, subset in enumerate(['train', 'valid']):
            with open('%s/%s/%s_log.csv' % (params['result_type_dir'], metric, subset), 'a', newline='') as file:
                writer = csv.writer(file)
                for row in loops_params['rows_by_metric'][metric_idx][subset_idx]:
                    writer.writerow(row)


def write_test_log(params, rows_by_metric):
    for metric_idx, metric in enumerate(es.metrics):
        with open('%s/%s/test_log.csv' % (params['result_type_dir'], metric), 'a', newline='') as file:
            writer = csv.writer(file)
            for row in rows_by_metric[metric_idx]:
                writer.writerow(row)


def log_progress(params, loops_params, timers_params, seeds_params, verbose=True):
    runtime_string = get_time_string(loops_params['runtime'])
    eta = get_time_string(sum(timer.eta for timer in timers_params['timers']))
    iteration = sum([timer.curr_iteration for timer in timers_params['timers']])
    progress = '%d/%d\tmod=%s \tseed=%d/%d \tinner_seed=%d/%d \tuser=%d/%d \ttime=%s \tETA=%s' % \
               (iteration, timers_params['iterations'], loops_params['mod_str'],
                seeds_params['seed_idx'] + 1, len(params['seeds']), seeds_params['inner_seed_idx'] + 1,
                len(params['inner_seeds']), loops_params['user_count'], params['num_users_to_test'], runtime_string,
                eta)
    with open('%s/progress_log.txt' % params['result_type_dir'], 'a') as file:
        file.write('%s\n' % progress)
    if verbose:
        print(progress)


def show_result(params):
    if es.make_tradeoff_plots:
        log_dir = '%s/%s' % (params['result_type_dir'], es.metrics[0])
        if len(es.model_names):
            AnalyseResults.binarize_results_by_compat_values(log_dir, 'test', len(params['diss_weights']) * 4,
                                                             print_progress=False)
            models_for_plotting = AnalyseResults.get_model_dict('jet')
            AnalyseResults.plot_results(log_dir, es.dataset_name, models_for_plotting, 'test_bins', True,
                                        show_tradeoff_plots=es.show_tradeoff_plots, diss_labels=False,
                                        performance_metric=es.metrics[0])
        else:  # only h1
            df = pd.read_csv('%s/test_log.csv' % log_dir)
            print(np.average(df['h1_acc'], weights=df['len']))


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


def main():
    params = dp.DataPreparations().get_experiment_parameters()
    timers_params = {}
    set_timers(params, timers_params)
    ncv.NestedCrossValidationProcess(params, timers_params)
    show_result(params)


if __name__ == "__main__":
    main()
