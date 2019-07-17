Code for the CheXaid paper

In order to train the model, the train data directory and other hyperparameters (batch size, fold number, etc.) can be set in `util/hypersearch.py`. After properly setting them, the training script can be run as:

```
python train.py
```

Alternatively, hyperparameters and file settings can be set by adding arguments to the training command as shown below.

```
python train.py \
       --lr 0.01

```
