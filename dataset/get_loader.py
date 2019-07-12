import torch.utils.data as data

from .concat_dataset import ConcatDataset
# from .su_dataset import SUDataset
# from .nih_dataset import NIHDataset
# from .tcga_dataset import TCGADataset
from .cxr_dataset import CXRDataset
from .label_mapper import TASK_SEQUENCES
from .pad_collate import PadCollate


def get_loader(data_args,
               transform_args,
               split,
               task_sequence,
               su_frac,
               nih_frac,
               cxr_frac,
               tcga_frac,
               batch_size,
               is_training=False,
               shuffle=False,
               study_level=False,
               frontal_lateral=False,
               return_info_dict=False,
               covar_list='',
               fold_num=None):

    """Returns a dataset loader.

       If both stanford_frac and nih_frac is one, the loader
       will sample both NIH and Stanford data.

    Args:
        su_frac: Float that specifies what percentage of stanford to load.
        nih_frac: Float that specifies what percentage of NIH to load.
        cxr_frac: Dictionary that specifies what fraction of each CXR dataset is needed.
        # TODO: remove all the frac arguments and instead pass a dictionary
        split: String determining if this is the train, valid, test, or sample split.
        shuffle: If true, the loader will shuffle the data.
        study_level: If true, creates a loader that loads the image on the study level.
            Only applicable for the SU dataset.
        frontal_lateral: If true, loads frontal/lateral labels.
            Only applicable for the SU dataset.
        return_info_dict: If true, return a dict of info with each image.
        covar_list: List of strings, specifying the covariates to be sent along with the images. 

    Return:
        DataLoader: A dataloader
    """

    if is_training:
        study_level = data_args.train_on_studies

    datasets = []
    for cxr_ds in ['pocus', 'hocus', 'pulm']:
        if cxr_ds in cxr_frac.keys() and cxr_frac[cxr_ds] != 0:
            if cxr_ds == 'pocus':
                data_dir = data_args.pocus_data_dir
                img_dir = None
            elif cxr_ds == 'hocus':
                data_dir = data_args.hocus_data_dir
                img_dir = None
            else:
                data_dir = data_args.pulm_data_dir
                img_dir = data_args.pulm_img_dir

            datasets.append(
                CXRDataset(
                    data_dir,
                    transform_args, split=split,
                    covar_list=covar_list,
                    is_training=is_training,
                    dataset_name=cxr_ds,
                    tasks_to=task_sequence,
                    frac=cxr_frac[cxr_ds],
                    toy=data_args.toy,
                    img_dir=img_dir,
                    fold_num=fold_num,
                )
            )

    if len(datasets) == 2:
        assert study_level is False, "Currently, you can't create concatenated datasets when training on studies"
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]

    # Pick collate function
    if study_level and not data_args.eval_tcga:
        collate_fn = PadCollate(dim=0)
        loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8,
                             collate_fn=collate_fn)
    else:
        loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8)

    return loader


def get_eval_loaders(data_args, transform_args, task_sequence, batch_size,
                     frontal_lateral, return_info_dict=False, covar_list='', fold_num=None):
    """Returns a dataset loader
       If both stanford_frac and nih_frac is one, the loader
       will sample both NIH and Stanford data.

    Args:
        eval_su: Float that specifes what percentage of stanford to load.
        nih_frac: Float that specifes what percentage of NIH to load.
        args: Additional arguments needed to load the dataset.
        return_info_dict: If true, return a dict of info with each image.
        covar_list: List of covariates as strings that we want from the dataset

    Return:
        DataLoader: A dataloader

    """

    eval_loaders = []

    
    if data_args.eval_pocus or data_args.eval_hocus or data_args.eval_pulm:
        for phase in ['train', 'valid']:
            eval_loaders += [get_loader(data_args,
                                        transform_args,
                                        phase,
                                        task_sequence,
                                        su_frac=0,
                                        nih_frac=0,
                                        cxr_frac={data_args.dataset_name: 1},
                                        tcga_frac=0,
                                        batch_size=batch_size,
                                        is_training=False,
                                        shuffle=False,
                                        return_info_dict=return_info_dict,
                                        covar_list=covar_list,
                                        fold_num=fold_num)]

    return eval_loaders

