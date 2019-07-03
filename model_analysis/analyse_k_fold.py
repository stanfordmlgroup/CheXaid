import numpy as np
import json
import os
import re
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import copy
import collections
from collections import defaultdict

PREFIX = ''
CKPT_DIR = '/deep/u/pranavsr/aihc-fall18-2dimaging/ckpts/'

# Regular expressions to match lines with
accuracy_regex = r'(?:accuracy_)([01].[0-9]+)?'
auc_regex = r'(?:AUROC_)([01].[0-9]+)?'

class ModelData(object):
    """Store model data - args, performance, optimal performance conditions."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.args = {}
        self.ckpt_paths = []
        self.performances = []

    def get_metric_value(self, ckpt_path, metric_regex):
        match = re.search(metric_regex, ckpt_path)
        if match is not None:
            return float(match.group(1))

    def get_best_valid_performance(self, metric_regex):
        performances = []
        for ckpt_path in self.ckpt_paths:
            metric_value = self.get_metric_value(ckpt_path, metric_regex)
            if metric_value is not None:
                performances.append((metric_value, ckpt_path))
        return max(performances, key=lambda item:item[0]) if len(performances) > 0 else (0, None)

    def add_performance(self, performance_instance):
        self.performances.append(performance_instance)

    def add_args_obj(self, args_path):
        with open(args_path, 'r') as args_file:
            self.args = json.loads(args_file.read())


def populate_models():
    """Populate and return all the model classes according to the ckpts."""

    models = {}
    # Document the performance of all models in CKPT_DIR
    for model_dir in os.listdir(PREFIX + CKPT_DIR):
        try:
            model_name = str(model_dir)
            new_model = ModelData(model_name)

            # Save args file for a given model
            new_model.add_args_obj(PREFIX + CKPT_DIR + model_dir + '/args.json')

            # Save all iterations of a model
            for iteration_ckpt in os.listdir(PREFIX + CKPT_DIR + model_dir):
                if iteration_ckpt.endswith(('.tar')):
                    new_model.ckpt_paths.append(str(iteration_ckpt))
            models[model_name] = new_model
        except:
            continue
    return models

def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary: transform from dict of dict to period separated keys.

    Args:
        d (dict): a dictionary to flatten.
        sep (str): a string to separate the keys.

    For example, if we had the following:
    d =
    {
        optim_args: {
            lr: 0.0001
        }
    }
    to:
    d =
    {
        optim_args.lr: 0.0001
    }
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def compare_jsons(json1, json2,
        ignore_fields=['data_args.fold_num', 'logger_args.dir_name', 'model_args.ckpt_path', 'num_epochs', 'logger_args.save_dir']):
    """Compares two json objects and return the keys that differ.

    Args:
        ignore_fields (list): list of field names to ignore in comparison.

    Returns:
        bool: indicating whether or not the jsons are identical (excluding ignored values).
        list: names of fields that differ between the jsons.
    """
    json1_copy = copy.deepcopy(json1)
    json2_copy = copy.deepcopy(json2)
    flattened_dict_1 = flatten(json1_copy)
    flattened_dict_2 = flatten(json2_copy)

    # Set ignored values to None (differences between these fields are ignored).
    for field in ignore_fields:
        flattened_dict_1[field] = None
        flattened_dict_2[field] = None

    # Compare dictionaries and find different fiels
    all_keys = set(flattened_dict_1.keys()) & set(flattened_dict_2.keys())
    diff_keys = []
    for key in all_keys:
        if flattened_dict_1[key] != flattened_dict_2[key]:
            diff_keys.append(key)

    return len(diff_keys) == 0, diff_keys


def group_models_by_args(models):
    """Create groups according to the different args.

    Args:
        models (dict): mapping of model name to the model class.

    Returns:
        dict: groupings of models. Maps model class to all the other classes that
            have the same params in args.
        list: names of all the keys that differ
    """
    groupings = defaultdict(list)
    all_diff_keys = []
    for model_name, model_class in models.items():
        model_args_json = model_class.args
        found_group = False
        for group_model, group_list in groupings.items():
            group_args_json = group_model.args
            is_same_json, diff_keys = compare_jsons(model_args_json, group_args_json)
            if is_same_json:
                group_list.append(model_class)
                found_group = True
                break
            else:
                all_diff_keys += diff_keys

        if not found_group:
            groupings[model_class].append(model_class)
    return groupings, set(all_diff_keys)

def add_args(csv_dict, keys, model_class):
    """Add args to csv_dict (corresponding to keys, taken from model class).

    The function updates csv dict. It iterates over all the keys, get the corresponding
    arg value from the model_class, and adds the value to the csv_dict.
    """
    for diff_key in keys:
        diff_key_list = diff_key.split('.')
        curr_args = model_class.args
        for key in diff_key_list:
            if curr_args and key in curr_args:
                curr_args = curr_args[key]
            else:
                curr_args = None
        csv_dict[diff_key_list[-1]].append(curr_args)

def get_different_arg_names(diff_keys):
    return [diff_key.split('.')[-1] for diff_key in diff_keys]

def add_summary_values(csv_dict, different_keys, performance_column_name='ckpt_performance'):
    """Add an entry to the csv that summarizes the performance of each arg across different values."""
    performances = csv_dict[performance_column_name]
    for column_name_full in different_keys:
        column_name = column_name_full.split('.')[-1]
        if column_name in csv_dict.keys():
            column_value_to_performance = defaultdict(list)
            column_values = csv_dict[column_name]
            for i, column_value in enumerate(column_values):
                column_value_to_performance[column_value].append(performances[i])
            result_string = ''
            for value, performance_list in column_value_to_performance.items():
                performance_mean = np.mean(performance_list)
                if len(result_string) > 0:
                    result_string += ', '
                result_string += f'{value}: {performance_mean}'
            csv_dict[column_name].append(result_string)

    # This just adds dummy values to keep lists in similar length for pandas.
    longest_list = np.max([len(value) for key, value in csv_dict.items()])
    for key, value_list in csv_dict.items():
        num_values_to_add = longest_list - len(value_list)
        if num_values_to_add > 0:
            value_list += [''] * num_values_to_add


def produce_csv_report(models, metric_regex):
    """Produce and save a csv summarizing performance of the different models."""
    model_groupings, all_diff_keys = group_models_by_args(models)

    csv_dict = defaultdict(list)
    # For a group of models with the same parameters
    group_number = 0
    for group_class, group_list in model_groupings.items():
        group_performances = []
        good_lists_count = 0
        for model_class in group_list:
            best_performance, best_ckpt = model_class.get_best_valid_performance(metric_regex)
            if best_ckpt:
                good_lists_count += 1
                group_performances.append(best_performance)
                csv_dict['ckpt_performance'].append(best_performance)
                csv_dict['group_number'].append(group_number)
                csv_dict['ckpt_path'].append(model_class.model_name + '/' + best_ckpt)
                csv_dict['fold_num'].append(model_class.args['data_args']['fold_num'])
                add_args(csv_dict, all_diff_keys, model_class)

        if len(group_performances) > 0:
            group_max, group_mean = np.max(group_performances), np.mean(group_performances)
            csv_dict['group_max'] += [group_max] * good_lists_count
            csv_dict['group_mean'] += [group_mean] * good_lists_count
            group_number += 1

    # Add a line that summarizes the performance across values of each argument.
    add_summary_values(csv_dict, all_diff_keys)

    # Format df and write csv
    df = pd.DataFrame.from_dict(csv_dict)
    df = df.sort_values(by=['group_number', 'ckpt_performance'], ascending=[True, False])
    diff_key_names = get_different_arg_names(all_diff_keys)
    column_order = ['group_number', 'ckpt_performance', 'fold_num', 'group_max', 'group_mean', 'ckpt_path'] + diff_key_names
    df = df[column_order]
    df.to_csv('k_fold_results.csv', index=False)


if __name__ == '__main__':
    models = populate_models()
    if len(models.keys()) == 0:
        print(f'No models found in {CKPT_DIR}')
    else:
        produce_csv_report(models, auc_regex)



