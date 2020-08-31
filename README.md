Code for `CheXaid: Deep Learning Assistance for Physician Diagnosis of Tuberculosis using Chest X-Rays in Patients with HIV`.

This is a deep learning model developed by Stanford Machine Learning Group (stanfordmlgroup.github.io) to help physicians detect Tuberculosis (TB) among patients with HIV infection using their Chest X-rays. 

## Data
By default, the training data directory is set to be `dummy/`, which has 128 randomly generated images. You can replace it with your own data.

## Prerequisites
This code uses python and pytorch. Follow these steps to get started:

1. Create a conda virtual environment
`conda env create -f environment.yml`

2. Activate the virtual environment
`conda activate chexaid`

3. If using dummy data: create dummy data dataframe
`python create_dummy_df.py`

## Training

To train a model with the default parameters run:
```
python train.py
```

#### Changing trainging parameters
By default, hyperparameters for training are set in `util/hypersearch.py`. This script includes information about the pretrained model to use for transfer learning (checkpoint paths, number of original classes), data args (task sequence, number of the fold to train and validate on for k-fold cross validation, image transformations), model args (model to be used, list of covariates, loss function), optimizer args and logger args. More details about these can be found in `args/`.

Alternatively, hyperparameters can be set by adding arguments to the training command:
```
python train.py \
       --batch_size=64 \
       --eval_pulm=True \
       --num_epochs=10
```

## Evaluating model
In order to test a model that has been trained as above, the test script can be run as:

```
python test.py \
       --eval_pulm=True \
       --transform_classifier=False \
       --save_cams=True \
       --task_sequence=pulm \
       --ckpt_path=ckpts/new_gt_1549918027265_4CF892/iter_6656_pulm-valid_pulm_tbAUROC_0.79.pth.tar \
       --split=test
```

Note that the `ckpt_path` argument should point to the checkpoints of the evaluated model. By default, model checkpoints will be saved following the format: `ckpts/new_gt_<timestamp>/<checkpoint name>.pth.tar`


