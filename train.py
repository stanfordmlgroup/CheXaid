import torch
import torch.nn as nn
import pandas as pd

import util
import models
from args import TrainArgParser
from eval import get_evaluator
from eval.loss import get_loss_fn
from logger import TrainLogger
from saver import ModelSaver
from dataset import get_loader, get_eval_loaders
from dataset import TASK_SEQUENCES


def train(args):
    """Run training loop with the given args.

    The function consists of the following steps:
        1. Load model: gets the model from a checkpoint or from models/models.py.
        2. Load optimizer and learning rate scheduler.
        3. Get data loaders and class weights.
        4. Get loss functions: cross entropy loss and weighted loss functions.
        5. Get logger, evaluator, and saver.
        6. Run training loop, evaluate and save model periodically.
    """

    model_args = args.model_args
    logger_args = args.logger_args
    optim_args = args.optim_args
    data_args = args.data_args
    transform_args = args.transform_args

    task_sequence = TASK_SEQUENCES[data_args.task_sequence]
    print('gpus: ', args.gpu_ids)
    # Get model
    if model_args.ckpt_path:
        model_args.pretrained = False
        model, ckpt_info = ModelSaver.load_model(model_args.ckpt_path, args.gpu_ids, model_args, data_args)
        if not logger_args.restart_epoch_count:
            args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[model_args.model]
        model = model_fn(task_sequence, model_args)
        num_covars = len(model_args.covar_list.split(';'))
        model.transform_model_shape(len(task_sequence), num_covars)
        if model_args.hierarchy:
            model = models.HierarchyWrapper(model, task_sequence)
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    optimizer = util.get_optimizer(model.parameters(), optim_args)
    lr_scheduler = util.get_scheduler(optimizer, optim_args)

    # The optimizer is loaded from the ckpt if one exists and the new model
    # architecture is the same as the old one (classifier is not transformed).
    if model_args.ckpt_path and not model_args.transform_classifier:
        ModelSaver.load_optimizer(model_args.ckpt_path, args.gpu_ids, optimizer, lr_scheduler)

    # Get loaders and class weights
    train_csv_name = 'train'
    if data_args.uncertain_map_path is not None:
        train_csv_name = data_args.uncertain_map_path

    # Put all CXR training fractions into one dictionary and pass it to the loader
    cxr_frac = {'pocus': data_args.pocus_train_frac, 'hocus': data_args.hocus_train_frac,
                'pulm': data_args.pulm_train_frac}
    train_loader = get_loader(data_args,
                              transform_args,
                              train_csv_name,
                              task_sequence,
                              data_args.su_train_frac,
                              data_args.nih_train_frac,
                              cxr_frac,
                              data_args.tcga_train_frac,
                              args.batch_size,
                              frontal_lateral=model_args.frontal_lateral,
                              is_training=True,
                              shuffle=True,
                              covar_list=model_args.covar_list,
                              fold_num=data_args.fold_num)
    eval_loaders = get_eval_loaders(data_args,
                                    transform_args,
                                    task_sequence,
                                    args.batch_size,
                                    frontal_lateral=model_args.frontal_lateral,
                                    covar_list=model_args.covar_list,
                                    fold_num=data_args.fold_num)
    class_weights = train_loader.dataset.class_weights

    # Get loss functions
    uw_loss_fn = get_loss_fn(args.loss_fn, args.device, model_args.model_uncertainty,
        args.has_tasks_missing, class_weights=class_weights)
    w_loss_fn = get_loss_fn('weighted_loss', args.device, model_args.model_uncertainty,
        args.has_tasks_missing, class_weights=class_weights)

    # Get logger, evaluator and saver
    logger = TrainLogger(logger_args, args.start_epoch, args.num_epochs, args.batch_size,
        len(train_loader.dataset), args.device, normalization=transform_args.normalization)
    
    eval_args = {}
    eval_args['num_visuals'] = logger_args.num_visuals
    eval_args['iters_per_eval'] = logger_args.iters_per_eval
    eval_args['has_missing_tasks'] = args.has_tasks_missing
    eval_args['model_uncertainty'] = model_args.model_uncertainty
    eval_args['class_weights'] = class_weights
    eval_args['max_eval'] = logger_args.max_eval
    eval_args['device'] = args.device
    eval_args['optimizer'] = optimizer
    evaluator = get_evaluator('classification', eval_loaders, logger, eval_args)

    print("Eval Loaders: %d" % len(eval_loaders))
    saver = ModelSaver(**vars(logger_args))

    metrics = None
    lr_step = 0
    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        for inputs, targets, info_dict, covars in train_loader:
            logger.start_iter()

            # Evaluate and save periodically
            metrics, curves = evaluator.evaluate(model, args.device, logger.global_step)
            logger.plot_metrics(metrics)
            metric_val = metrics.get(logger_args.metric_name, None)
            assert logger.global_step % logger_args.iters_per_eval != 0 or metric_val is not None
            saver.save(logger.global_step, logger.epoch, model, optimizer, lr_scheduler, args.device,
                       metric_val=metric_val, covar_list=model_args.covar_list)
            lr_step = util.step_scheduler(lr_scheduler, metrics, lr_step, best_ckpt_metric=logger_args.metric_name)

            # Input: [batch_size, channels, width, height]

            with torch.set_grad_enabled(True):
            # with torch.autograd.set_detect_anomaly(True):

                logits = model.forward([inputs.to(args.device), covars])

                # Scale up TB so that it's loss is counted for more if upweight_tb is True.
                if model_args.upweight_tb is True:
                    tb_targets = targets.narrow(1, 0, 1)
                    findings_targets = targets.narrow(1, 1, targets.shape[1] - 1)
                    tb_targets = tb_targets.repeat(1, targets.shape[1] - 1)
                    new_targets = torch.cat((tb_targets, findings_targets), 1)

                    tb_logits = logits.narrow(1, 0, 1)
                    findings_logits = logits.narrow(1, 1, logits.shape[1] - 1)
                    tb_logits = tb_logits.repeat(1, logits.shape[1] - 1)
                    new_logits = torch.cat((tb_logits, findings_logits), 1)
                else:
                    new_logits = logits
                    new_targets = targets

                    
                unweighted_loss = uw_loss_fn(new_logits, new_targets.to(args.device))

                weighted_loss = w_loss_fn(logits, targets.to(args.device)) if w_loss_fn else None

                logger.log_iter(inputs, logits, targets, unweighted_loss, weighted_loss, optimizer)

                optimizer.zero_grad()
                if args.loss_fn == 'weighted_loss':
                    weighted_loss.backward()
                else:
                    unweighted_loss.backward()
                optimizer.step()

            logger.end_iter()

        logger.end_epoch(metrics, optimizer)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TrainArgParser()
    args = util.get_auto_args(parser)
    train(args)

