from sklearn import metrics
import torch
import numpy as np
from torchmetrics import AUROC,AveragePrecision
def f1(score,label):
    auc_score = metrics.roc_auc_score(label, score)
    aupr_sc = metrics.average_precision_score(label,score)
    print('auc',auc_score)
    print('aupr',aupr_sc)
def f2(score,label):
    score = torch.sigmoid(score)
    auroc = AUROC(pos_label=1)
    average_precision = AveragePrecision(pos_label=1)
    auc_score = auroc(score, label)
    aupr_sc = average_precision(score, label)
    print('auc', auc_score)
    print('aupr', aupr_sc)
def f3(score,label):
    max_score = np.max(score)
    score = score-max_score
    score = np.exp(score)/(1.+np.exp(score))
    auc_score = metrics.roc_auc_score(label, score)
    aupr_sc = metrics.average_precision_score(label, score)
    print('auc', auc_score)
    print('aupr', aupr_sc)
if __name__ == '__main__':

    score = torch.randn(100)
    label = torch.ones(100,dtype=torch.long)
    label[50:]=0
    f1(score.numpy(),label.numpy())
    f2(score,label)
    f3(score.numpy(),label.numpy())
