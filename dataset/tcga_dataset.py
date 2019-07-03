"""Dataset for whole slide images from TCGA data"""
from pathlib import Path
import os
import h5py
import numpy as np
import os
import pickle
import random
import torch.utils.data as data
import torch
import pandas as pd
from tqdm import tqdm
import util
from .base_dataset import BaseDataset
from .constants import COL_TCGA_SLIDE_ID, COL_TCGA_FILE_ID,\
        COL_TCGA_FILE_NAME, COL_TCGA_CASE_ID, COL_TCGA_LABEL,\
        COL_TCGA_PATCH_ID, COL_TCGA_NUM_PATCHES, COL_TCGA_INDICES,\
        COL_TCGA_PATH, COL_TCGA_ENTITIES, SLIDE_METADATA_FILE,\
        SLIDE_PKL_FILE, DEFAULT_PATCH_SIZE 

# TODO: implement flag to turn dataset into toy dataset
class TCGADataset(BaseDataset):
    """Dataset for TCGA classification."""

    def __init__(self, data_path, transform_args, metadata_csv,
                 split, num_classes=2,
                 resize_shape=(DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE),
                 max_patches=None, tasks_to='tcga',
                 is_training=False, filtered=True, toy=False):
        """Initialize TCGADataset.

        data directory to be organized as follows:
            data_path
                slide_list.pkl
                train.hdf5
                val.hdf5
                test.hdf5
                metadata.csv

        Args:
            data_path (str): path to data directory
            transform_args (args): arguments to transform data
            metadata_csv (str): path to csv containing metadata information of the dataset
            split (str): either "train", "valid", or "test"
            num_classes (int): number of unique labels
            resize_shape (tuple): shape to resize the inputs to
            max_patches (int): max number of patches to obtain for each slide
            tasks_to (str): corresponds to a task sequence
            is_training (bool): whether the model in in training mode or not
            filtered (bool): whether to filter the images
        """
        if split not in ["train", "valid", "test"]:
            raise ValueError("Invalid value for split. Must specify train, valid, or test.")

        super().__init__(data_path, transform_args,
                         split, is_training, 'tcga', tasks_to)
        self.data_path = data_path
        self.slide_list_path = os.path.join(self.data_path, SLIDE_PKL_FILE) 
        self.hdf5_path = os.path.join(self.data_path, "{}.hdf5".format(split))

        self.hdf5_fh = h5py.File(self.hdf5_path, "r")

        self.split = split
        self.is_training = is_training
        self.metadata_path = os.path.join(self.data_dir, metadata_csv)
        self.metadata = pd.read_csv(self.metadata_path)
        print("hdf5 path: {}".format(self.hdf5_path))

        self.toy = True

        self.filtered = filtered 
        with open(self.slide_list_path, "rb") as pkl_fh:
            self.slide_list = pickle.load(pkl_fh)

        self.valid_slides = [slide_id for slide_id in self.hdf5_fh]
        print("Num valid slides {}".format(len(self.valid_slides)))

        self.num_classes = num_classes

        self.resize_shape = resize_shape
        self.max_patches_per_slide = max_patches

        self.patch_list = self._get_patch_list()
        print("Patch list shape: {}".format(self.patch_list.shape))

        #TODO: Replace it with a function that calculates weights based on the current dataset
        self.class_weights = [[0.3, 0.4], [0.7, 0.6]]

    def get_patch(self, slide_id, patch_num):
        """Index into specific patch within slide"""
        patch = self.hdf5_fh[slide_id][int(patch_num)]
        patch = patch.astype(np.float32)
        return patch

    def __len__(self):
        return len(self.patch_list)

    def _label_conversion(self, label, label_dict=None):
        """Turn string label into integer"""
        if label_dict is None:
            label_dict = {"bladder": 0, "thyroid": 1}
        if label not in label_dict:
            raise ValueError("Invalid label: {} entered".format(label))
        return label_dict[label]

    def transform_label(self, label):
        """Make label correct shape"""
        label = np.array(label).astype(np.float32).reshape(1)
        return label

    def __getitem__(self, idx):
        """Return element of dataset"""
        slide_id, patch_num = self.patch_list[idx]
        label = self.get_slide_label(slide_id)
        label = self._label_conversion(label)
        label = self.transform_label(label)
        label = torch.tensor(label, dtype=torch.float32)
        patch = self.get_patch(slide_id, patch_num)
        # TODO: transform patch
        info_dict = {COL_TCGA_SLIDE_ID: slide_id,
                     COL_TCGA_PATCH_ID: patch_num}
        return patch, label, info_dict

    def get_slide_label(self, slide_id):
        """Get label for slide"""
        df = self.metadata.set_index(COL_TCGA_SLIDE_ID)
        return df.loc[slide_id, COL_TCGA_LABEL]

    def _all_patches(self, valid_slides, max_patches_per_slide):
        """Return shuffled list of (slide, patch_idx) tuples"""
        patch_list = [[(slide[COL_TCGA_SLIDE_ID], str(patch_idx))
                for patch_idx in range(len(self.hdf5_fh[slide[COL_TCGA_SLIDE_ID]]))]
                        for slide in valid_slides]
        if max_patches_per_slide is not None:
            patch_list = [random.sample(slide_patches, max_patches_per_slide)
                    for slide_patches in patch_list]
        patch_list = [tup for slide_patches in patch_list
                      for tup in slide_patches] 
        random.shuffle(patch_list)
        return np.array(patch_list)


    def _get_patch_list(self):
        """Return list of patches to be used for training.

        If patches are pre-filtered, simply add all patches from
        every slide. Otherwise, use is_patch_tumor to differentiate
        patches with tumor cells form patches without. 

        Returns:
            patch_list (list): list of (slide_id, patch_idx) tuples
        """
        if self.max_patches_per_slide is not None and self.max_patches_per_slide > 0:
            print(("Using at most {} patches per slide!").format(
                self.max_patches_per_slide))

        valid_slide_list = list(filter(
                lambda x: x[COL_TCGA_SLIDE_ID] in self.valid_slides, self.slide_list))

        print("Generating Patches...")
        patch_list = []
        if self.filtered: 
            return self._all_patches(valid_slide_list, self.max_patches_per_slide)

        for slide_idx, slide in enumerate(valid_slide_list):
            print("Processing slide: {}".format(slide[COL_TCGA_SLIDE_ID]))
            num_patches = slide[COL_TCGA_NUM_PATCHES]
             
            patch_list_in_slide = slide[COL_TCGA_INDICES] if COL_TCGA_INDICES in slide \
                    else list(range(num_patches))
            random.shuffle(patch_list_in_slide)
            count_added_to_patch_list = 0

            for patch_idx in patch_list_in_slide:
                if patch_idx % 1000 == 0:
                    print(patch_idx)
                label = self.get_slide_label(slide[COL_TCGA_SLIDE_ID])
                tup = (slide[COL_TCGA_SLIDE_ID], str(patch_idx))
                if 'indices' not in slide:
                    patch = self.get_patch(slide[COL_TCGA_SLIDE_ID], patch_idx)
                    patch= np.moveaxis(patch, 0, 2) / 255
                    if not util.is_patch_tumor(patch):
                        continue
                patch_list.append(tup)
                count_added_to_patch_list += 1
                if count_added_to_patch_list == self.max_patches_per_slide:
                    break

        return np.array(patch_list)
