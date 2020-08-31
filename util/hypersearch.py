import argparse
from argparse import Namespace
import sys
import util
import numpy as np
import os
import json
import random


def get_auto_args(parser):
    def get_data_args():
        pretrained_model_dict = {
                'ckpt_path': '',
                'transform_classifier': True,
                'n_orig_classes': 7,
                'model': 'DenseNet121',
            }

        data_args = {
            # data params
            'data_args.pulm_data_dir': './',
            'data_args.eval_pulm': True,
            'data_args.task_sequence': 'pulm',
            'data_args.fold_num': 0,

            'transform_args.normalization': 'cxr_pulm_tb_norm',
            'transform_args.scale': 320,
            'transform_args.crop': 320,
            'transform_args.transform_affine_all': 25,
            'transform_args.clahe': False,

            'data_args.nih_train_frac': 0,
            'data_args.su_train_frac': 0,
            'data_args.pulm_train_frac': 1.0,

            'batch_size': batch_size,

            'has_tasks_missing': False,
            'is_training': True,
            'start_epoch': 1,

             # model params
            'model_args.model': pretrained_model_dict['model'],

            'model_args.transform_classifier' : pretrained_model_dict['transform_classifier'],
            'model_args.n_orig_classes' : pretrained_model_dict['n_orig_classes'],
            'model_args.ckpt_path' : pretrained_model_dict['ckpt_path'],
            'model_args.covar_list': 'prev_tb;age;art_status;temp;oxy_sat;hgb;cd4;wbc',
            'model_args.upweight_tb': True,
            'loss_fn': 'focal_loss',

             # training params
            'logger_args.iters_per_print': int(32*batch_size),
            'logger_args.iters_per_eval':  int(32*batch_size),
            'logger_args.iters_per_save': int(32*batch_size),
            'logger_args.iters_per_visual': int(32*batch_size),
            'logger_args.metric_name': 'pulm-valid_pulm_tbAUROC',
            'logger_args.maximize_metric': True,
            'logger_args.restart_epoch_count': True,
         }
        return data_args

    def get_training_knobs():
        learning_rate = float(10.0 ** -4)
        weight_decay = float(10.0 ** -4)
        scheduler = 'step'
        training_knobs = {
            # training knobs
            'num_epochs': 20,
            'optim_args.lr': learning_rate,
            'optim_args.lr_scheduler': scheduler,
            'optim_args.weight_decay': weight_decay,
            'logger_args.name': f'new_gt',
        }

        if scheduler == 'step':
            lr_decay_step = 500
            scheduler_knobs = {
                'optim_args.lr_decay_step': lr_decay_step,
                'optim_args.lr_decay_gamma': 0.5,
            }
        else:
            raise NotImplementedError('Scheduler args not implemented')
        training_knobs.update(scheduler_knobs)
        return training_knobs

    def get_optimizer_knobs():
        if optimizer == 'adam':
            optimizer_knobs = {
                'optimizer': 'adam',
            }
        elif optimizer == 'sgd':
            momentum = 0.9
            sgd_dampening = 0
            optimizer_knobs = {
                # optimizer knobs
                'optim_args.optimizer': 'sgd',
                'optim_args.sgd_momentum': momentum,
                'optim_args.sgd_dampening': sgd_dampening,
            }
        else:
            raise NotImplementedError('Optimizer args not implemented')
        return optimizer_knobs

    batch_size = 16
    optimizer = 'adam'

    defaults = parser.parser.parse_args()
    dict_def = vars(defaults)
    data_args = get_data_args()
    dict_def.update(data_args)
    training_knobs = get_training_knobs()
    dict_def.update(training_knobs)
    optimizer_knobs = get_optimizer_knobs()
    dict_def.update(optimizer_knobs)
    ns = Namespace(**dict_def)
    args = parser.process_args(ns)
    print(args)

    return args
