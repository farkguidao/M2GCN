# M2GCN

 Source code for paper "M2GCN: Multi-Modal Graph Convolutional Network for Polypharmacy Side Effects Discovery"

## Requirements

The code has been tested under Python 3.8, with the following packages installed (along with their dependencies):

- torch >= 1.8.1
- pytorch-lightning >= 1.4.1
- torchmetrics >= 0.4.1
- numpy
- pandas
- tqdm
- yaml

## Files in the folder


- **/data:** Store the example inputs.
- **/dataloader:** Codes of the dataloader and the negative sampler.
- **/models:** Code of the M2GCN model.
- **/setting:** Store the hyperparameter configuration file.
- **/utils:** Codes for sparse matrix operation and calculating laplacian matrix.
- **/lightning_logs:** Store the trained model parameters, checkpoints, logs and results.
- **main.py:** The main entrance of running.

## Data

M2GCN expect an edgelist for the input network, i.e.,

```
node1 node2
node1 node3
...

```

Each row contains two IDs, which are separated by '\t' and mean there is an interaction between the two nodes.

There are five useful inputs files in this directory `/data`, namely:

- **drug_adj.txt:** Drug-drug interactions for training.
- **protein_adj.txt:** Protein-protein interactions for training.
- **dp_adj.txt:** Drug-protein interactions for training.
- **test_false.txt:** Drug-Drug interactions for testing (positive samples) .
- **test_true.txt:** Drug-Drug no interactions for testing (negative samples).

If you want to run our code directly, please ensure the consistency of the data file name, because we wrote it directly in the code of the model.

## Basic usage

### Train M2GCN

train M2GCN by

```
python main.py
```

The default configuration file is `setting/settings.yaml`.

And if you want to adjust the hyperparameters of the model, you can modify it in `.setting/settings.yaml`, or create a similar configuration file, and specify `--setting_path` like this:

```
python main.py --setting_path yourpath.yaml
```

Checkpoints, logs, and results during training will be stored in the directory: `./lightning_logs/version_0`

And you can run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress.

### Link Prediction with pre-trained model

You can predict the interaction between drugs through the pre-trained model we provide.

**Since git limits the size of a single file upload, we divide the pre-trained model into multiple volumes. Please unzip the files in the directory `./lightning_logs/pre-trained/checkpoints/` first.**

Load the pre-trained model and predict the test dataset by:

```
python main.py --test --ckpt_path ./lightning_logs/pre-trained/checkpoints/pre-trained.ckpt
```

The result(auc,aupr) will be stored in the directory: `./lightning_logs/version_0`

If you want to load your trained model to predict the test data set, you only need to change `--ckpt_path`like this:

```
python main.py --test --ckpt_path yourpath.ckpt
```

PS: Keep the configuration file unchanged during training and testing.
