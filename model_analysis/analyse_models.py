import numpy as np
import json
import os
import re
import pickle
import matplotlib.pyplot as plt

PREFIX = ''#'/mnt/deep'
# PREFIX = ''
CKPT_DIR = '/deep/u/amits/cxr/aihc-fall18-2dimaging/ckpts/' #'/deep/u/pranavsr/aihc-fall18-2dimaging/ckpts/'

# Regular expressions to match lines with
start_epoch = r'start of epoch [0-9]+'
valid_auroc = r'valid.*?tbAUROC: [01](.[0-9]+)?'
auroc = r'AUROC: [01](.[0-9]+)?'
valid_loss = r'valid_loss: [01](.[0-9]+)?'


class ModelData(object):
    """Store model data - args, performance, optimal performance conditions."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.args = {}
        self.performances = []

    def get_best_valid_auroc(self):
        return np.amax(self.performances[:, 1])

    def get_best_valid_loss(self):
        return np.amin(self.performances[:, 2])
    
    def calculate_performances(self, lines):
        cur_epoch, cur_valid_auroc, cur_valid_loss, cur_train_loss = 0, 0.0, 0.0, 0.0
        for line in lines:
            content = detect_line_intent(line)                    
            if 'start_epoch' in content:
                if cur_epoch != 0:
                    self.performances.append([cur_epoch, cur_valid_auroc, cur_valid_loss])
                cur_epoch = content['start_epoch']
            elif 'valid_auroc' in content:
                cur_valid_auroc = content['valid_auroc']
            elif 'valid_loss' in content:
                cur_valid_loss = content['valid_loss']
        self.performances = np.array(self.performances)


def detect_line_intent(line):
    """Detect intent of the given line.
    
    Here, we check if the current line starts an epoch, specified the validation 
    auroc or the validation loss of the current epoch.
    """

    content = {}
    match = re.search(start_epoch, line)
    if match is not None:
        content = {'start_epoch': int(match.group()[15:])}
    elif re.search(valid_auroc, line) is not None:
        match = re.search(auroc, line)
        content = {'valid_auroc': float(match.group()[7:])}
    elif re.search(valid_loss, line) is not None:
        match = re.search(valid_loss, line)
        content = {'valid_loss': float(match.group()[12:])}
    return content


def get_best_models(models, models_perf, metric, num=3):
    if metric == 'valid_auroc':
        models_sorted = sorted(models_perf.items(), key=lambda x: x[1])[::-1]
    elif metric == 'valid_loss':
        models_sorted = sorted(models_perf.items(), key=lambda x: x[1])
        
    # Print the top 3 models according to the specified metric
    for model in models_sorted[:num]:
        f = open('model_arg_files/' + model[0] + '_' + metric + '_' + str(model[1]) + '.json', 'w', encoding='utf-8')
        json.dump(models[model[0]].args, f, indent=4)
        f.close()
    return models_sorted


def print_models_over_perf(models, models_sorted, valid_auroc=None, valid_loss=None):
    """Return selected models as a list.
    
    Only one of the metric arguments must be not None, and models_sorted
    must be of the form [model_name, metric_value] for the chosen metric
    (sorted). Return selected_models, as a list of models having performance
    over/under a certain threshold.
    """

    selected_models = []
    if valid_auroc is not None:
        metric = 'valid_auroc'
        for model in models_sorted:
            if model[1] > valid_auroc:
                selected_models.append(model[0])
                f = open(model[0] + '_' + metric + '_' + str(model[1]) + '.json',
                         'w', encoding='utf-8')
                json.dump(models[model[0]].args, f, indent=4)
                f.close()
    elif valid_loss is not None:
        metric = 'valid_loss'
        for model in models_sorted:
            if model[1] < valid_loss:
                selected_models.append(model[0])
                f = open(model[0] + '_' + metric + '_' + str(model[1]) + '.json',
                         'w', encoding='utf-8')
                json.dump(models[model[0]].args, f, indent=4)
                f.close()
    return selected_models


def print_csv_with_params(models, selected_models):
    """Record the performances of the selected models in a csv file."""
    
    with open('model_summary.csv', 'w') as csv_file:
        # Print the list of columns in the csv file
        csv_file.write(
            'Model_Name,dataset,best_val_auroc,best_val_loss,learning_rate,lr_decay_step,weight_decay,color_jitter,crop,scale,covar_list,optimizer,data_path\n')
        for model_name in selected_models:
            model = models[model_name]
            args = model.args
            color_jitter = args['transform_args']['color_jitter']
            crop = args['transform_args']['crop']
            lr = args['optim_args']['lr']
            weight_decay = args['optim_args']['weight_decay']
            lr_decay_step = args['optim_args']['lr_decay_step']
            dataset = args['data_args']['pulm_data_dir'].split("/")[-1]
            scale = args['transform_args']['scale']
            covar_list = args['model_args']['covar_list'] if 'covar_list' in args['model_args'] else None
            data_path = args['data_args']['pulm_data_dir']
            optimizer = args['optim_args']['optimizer']

            best_valid_auroc = model.get_best_valid_auroc()
            best_valid_loss = model.get_best_valid_loss()
            csv_file.write(model_name + ',' + dataset + ',' + str(best_valid_auroc) + ',' +
                           str(best_valid_loss) + ',' + str(lr) + ',' + str(lr_decay_step) +
                           ',' + str(weight_decay) + ',' + str(color_jitter) + ',' + str(crop) +','
                           + str(scale) + ',' + str(covar_list) + ',' + str(optimizer) +
                           ',' + str(data_path) + '\n')
            

def test_analysis_tools(models):
    """Show how to use this script. Change the params and CKPT_DIR to analyse 
    another set of models."""
    
    models_perf = {}
    for model in models:
        models_perf[model] = models[model].get_best_valid_auroc()

    # Get the sorted models as models_sorted and also store top 3 to file
    models_sorted = get_best_models(models, models_perf, 'valid_auroc', num=3)

    # Get all models over a specific threshold of metric value as selected_models and
    # store them to file
    selected_models = print_models_over_perf(models, models_sorted, valid_auroc=0.65)

    # print a csv file with all the relevant params for the selected models
    print_csv_with_params(models, selected_models)
    

if __name__ == '__main__':
    models = {}
    models_perf = {}
    
    # Document the performance of all models in CKPT_DIR
    for model_dir in os.listdir(PREFIX + CKPT_DIR):
        model_name = str(model_dir)
        new_model = ModelData(model_name)
        with open(PREFIX + CKPT_DIR + model_dir + '/args.json', 'r') as args_file:
            new_model.args = json.loads(args_file.read())

        log_file_name = None
        if 'sgd.log' in os.listdir(PREFIX + CKPT_DIR + model_dir):
            log_file_name = PREFIX + CKPT_DIR + model_dir + '/sgd.log'
        elif 'adam.log' in os.listdir(PREFIX + CKPT_DIR + model_dir):
            log_file_name = PREFIX + CKPT_DIR + model_dir + '/adam.log'
        if log_file_name:
            with open(log_file_name, 'r') as perf:
                print(model_dir)
                lines = perf.readlines()
                new_model.calculate_performances(lines)
                if len(new_model.performances) > 0:
                    models[model_name] = new_model
                    
    # Find the best models among all the models in CKPT_DIR
    test_analysis_tools(models)
