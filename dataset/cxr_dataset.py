from PIL import Image
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.constants import CXR_COV_MEAN, CXR_COV_STD
from .base_dataset import BaseDataset
from .label_mapper import TASK_SEQUENCES


class CXRDataset(BaseDataset):

    def __init__(self, data_dir, transform_args, split, covar_list, is_training,
                 dataset_name, tasks_to, frac, toy=False, img_dir=None, fold_num=None):
        """"Dataset class for all CXR datasets from Cape Town

        Args:
              data_dir (string): Name of the root data directory.
              transform_args (Namespace): Args for data transforms.
              split (string): Train, test or valid.
              covar_list(list of strings): List of covariates to include in the model
              dataset_name (string): Dataset to load (currently supported: hocus (Tom's) and pocus (Neil's)).
              is_training (bool): Indicate whether this is needed for training or not.
              tasks_to (dict): The sequence of tasks.
              frac (float): Fraction of training data to use.
              toy (bool): Indicate if only toy data is needed.
              img_dir (string): Path of the root data directory
        """

        super().__init__(data_dir, transform_args, split,
                         is_training, dataset_name, tasks_to)

        if img_dir is None:
            self.img_dir = self.data_dir / "images"
        else:
            self.img_dir = Path(img_dir)

        if len(covar_list) > 0:
            self.covar_list = covar_list.split(';')
        else:
            self.covar_list = ''

        self.study_level = False
        self.frontal_lateral = False
        self.fold_num = fold_num

        # load data from csv
        self.df = self.load_df()

        if frac != 1 and is_training:
            self.df = self.df.sample(frac=frac)
            self.df.reset_index(drop=True)

        self.subjids = self.df['subjID']

        self.labels = self.get_disease_labels()
        self.img_paths = self.get_paths()
        self.covariates = self.get_covariates()
        self._set_class_weights(self.labels)

    def load_df(self):
        """Load the data from data_dir to a Pandas dataframe."""
        if self.split == 'test':
            # there is no fold
            filename = (self.split + "_data.csv")
        else:
            filename = (self.split + "_data_" + str(self.fold_num)
                    + ".csv" if self.fold_num is not None else
                    self.split + "_data.csv")
        csv_path = self.data_dir / filename
        df = pd.read_csv(csv_path)
        df.reset_index(drop=True)
        # self.df is returned just to maintain clarity in the __init__
        # as to when df was created and altered
        return df

    def get_paths(self):
        """Get list of paths to images.

        Return:
            list: List of paths to images.
        """
        path_list = self.df['Path'].tolist()
        return path_list

    def get_disease_labels(self):
        """Return labels as K-element arrays for the three diseases.

        Return:
            ndarray: (N * K) numpy array of labels.
        """
        # construct the label matrix
        num_data_points = len(self.df.index)
        num_labels = len(TASK_SEQUENCES[self.dataset_name])
        labels = np.zeros([num_data_points, num_labels])

        # populate the label matrix
        diseases = [dis for dis in TASK_SEQUENCES[self.dataset_name]]
        label_df = self.df[diseases].apply(pd.to_numeric, errors='coerce')
        # remove all NaNs that came up because of the above operation
        label_df.fillna(0.0)

        for dis in diseases:
            label_df[dis] = (label_df[dis] >= 1).astype(int)

        for i, disease in enumerate(diseases):
            labels[:, i] = label_df[disease].tolist()

        return labels

    def get_covariates(self):
        """Return covariates for each data item as a matrix

        Return:
            ndarray: (N * C) numpy array of C covariates for N data points.
        """
        if len(self.covar_list) == 0:
            return ''

        # Construct the covariates matrix
        num_data_points = len(self.df.index)
        num_covars = len(self.covar_list)
        print("number of covariates: " + str(num_covars))
        covariates = np.zeros([num_data_points, num_covars])

        # Populate the covariates matrix
        covar_df = self.df[self.covar_list].apply(pd.to_numeric, errors='coerce')
        for column_name, column in covar_df.items():
          covar_df[column_name] = covar_df[column_name].replace(r'\s+', np.nan, regex=True)
          covar_df[column_name] = covar_df[column_name].fillna(0.0)
        # covar_df.fillna(0.0)

        for col_name, values in covar_df.items():
            if col_name in CXR_COV_MEAN:
                covar_df[col_name] = ((covar_df[col_name] - CXR_COV_MEAN[col_name]) / CXR_COV_STD[col_name])
        covariates = covar_df.values

        # print(covariates)

        return covariates

    def __getitem__(self, index):

        label = self.labels[index, :]
        subjID = self.subjids[index]
        if len(self.covariates) > 0:
            covars = self.covariates[index, :]
        else:
            covars = ''

        if self.label_mapper is not None:
            label = self.label_mapper.map(label)
        label = torch.FloatTensor(label)

        # Get and transform the image
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.transform(img)
        return img, label, {'subjID': subjID}, covars
