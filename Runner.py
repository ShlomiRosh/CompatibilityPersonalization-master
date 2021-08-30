import csv
from time import time

import DataPreparation as dp
import ExperimentSettings as es
import NestedCrossValidation as ncv


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


if __name__ == "__main__":
    main()
