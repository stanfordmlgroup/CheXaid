Code for the CheXaid paper

In order to train the model, the train data directory and other hyperparameters can be set in `util/hypersearch.py`. This includes information about the pretrained model to use for transfer learning (checkpoint paths, number of original classes), data args (task sequence, number of the fold to train and validate on for k-fold cross validation, image transformations), model args (model to be used, list of covariates, loss function), optimizer args and logger args. More details about these can be found in `args/`.


After properly setting them, the training script can be run as:

```
python train.py
```

Alternatively, hyperparameters and file settings can be set by adding arguments to the training command:

```
python train.py \
       --batch_size 64 \
       --eval_pulm True \
       --num_epochs 10

```

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

Note that `split=True` is necessary to run the script on the test data, and `save_cams=True` makes sure that the class activation maps of the model on the test data are saved. 