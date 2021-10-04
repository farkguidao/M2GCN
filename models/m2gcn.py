from typing import Optional, Union, List

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
import torchmetrics.functional
from utils.torch_sparse_utils import *
class M2GCNModel(pl.LightningModule):
    def __init__(self,N,adj_list,node_num_list,feature_dim_list,K=4,hidden_dim=64,em_dim=32,lam=0.01,pos_weight=1.,
                 learning_rate=0.01,weight_decay=0.0001,dropout=0.1):
        '''
        :param N: 总节点数,已弃用
        :param adj_list: 子图的邻接矩阵列表,稀疏矩阵
        :param node_num_list: node_num
        :param feature_dim_list: 特征维度
        :param K: gcn层数
        :param hidden_dim: 隐藏维度
        :param em_dim: 编码维度
        :param learning_rate: 学习率
        :param weight_decay: 权重衰减
        '''
        super(M2GCNModel, self).__init__()
        self.save_hyperparameters('N','K','node_num_list','feature_dim_list','hidden_dim','em_dim','lam','pos_weight','learning_rate','weight_decay','dropout')

        # 工程类组件

        # loss
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        # 模型组件
        self.T = len(adj_list)

        # inputs 初始特征
        self.inputs_list = nn.ParameterList()
        # 特征映射
        self.w_list = nn.ModuleList()

        for node_num, feature_dim in zip(node_num_list, feature_dim_list):
            inputs = nn.Parameter(torch.FloatTensor(node_num, feature_dim))
            nn.init.xavier_uniform_(inputs)
            self.inputs_list.append(inputs)
            self.w_list.append(nn.Linear(feature_dim, hidden_dim, bias=False))

        # GCN
        self.gcn_list = nn.ModuleList()
        # mlp
        self.mlp_list = nn.ModuleList()
        for i in range(self.T):
            self.gcn_list.append(GCN(N,adj_list[i],K))
            self.mlp_list.append(nn.Sequential(nn.Linear(hidden_dim,hidden_dim),
                                               nn.ReLU(),
                                               nn.Dropout(dropout),
                                               nn.Linear(hidden_dim, em_dim),
                                               nn.ReLU(),
                                               nn.Dropout(dropout)
                                               ))
        # mk
        self.mk = nn.Parameter(torch.FloatTensor(self.T,em_dim))
        nn.init.xavier_uniform_(self.mk)

        # save the best result
        self.best_auc = 0
        self.best_aupr = 0


    #-----------------工程代码--------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ 20, 30, 40,50],
        #                                                     gamma=0.1)
        # return [optimizer],[lr_scheduler]
        return optimizer



    def training_step(self, batch,batch_id):
        x,label = batch
        score,embedings = self(x)
        edge_type = x[:, 2]
        loss1 = 0
        for i in range(self.T):
            index = edge_type == i
            loss1 = loss1+self.loss(score[index],label[index])
        consis_loss = self.cal_consis_loss(embedings)
        loss = loss1+self.hparams.lam * consis_loss
        self.log('loss1',loss1,prog_bar=True)
        self.log('loss2',consis_loss,prog_bar=True)
        self.log('loss_all',loss,prog_bar=True)

        return loss


    def validation_step(self, batch,batch_id):
        x, label = batch
        score, _ = self(x)
        self.evaluate(score,label)

    def evaluate(self,score,label):
        score = torch.sigmoid(score)
        label=label.int()
        auc = torchmetrics.functional.auroc(score,label,pos_label=1)
        aupr = torchmetrics.functional.average_precision(score,label,pos_label=1)
        if auc>self.best_auc:
            self.best_auc = auc
            self.best_aupr = aupr
        self.log('auc',auc,prog_bar=True)
        self.log('aupr',aupr,prog_bar=True)


    def test_step(self, batch,batch_id):
        return self.validation_step(batch,batch_id)

    # 结束时存储最优结果
    def on_fit_end(self) -> None:
        with open(self.trainer.log_dir + '/best_result.txt', mode='w') as f:
            result = {'auc': float(self.best_auc), 'aupr': float(self.best_aupr)}
            print('best_result:', result)
            f.write(str(result))

    # -----------------核心代码-----------------------#

    def cal_node_embeding(self):
        # 特征映射
        inputs = []
        for f, w in zip(self.inputs_list, self.w_list):
            inputs.append(w(f))
        inputs = torch.cat(inputs)
        # 计算编码
        em_list=[]
        for i in range(self.T):
            em = self.gcn_list[i](inputs) # N,feature_dim
            em = self.mlp_list[i](em) # N,em_dim
            em_list.append(em)
        return torch.stack(em_list) # T,N,em_dim


    def cal_consis_loss(self,embedings):
        # embedings = torch.exp(embedings)
        # if self.global_step%10==0:
        #     self.mean = embedings.mean(dim=0).detach()
        mean = embedings.mean(dim=0) #N,em_dim
        mean = mean.detach()
        consis_loss = torch.mean((embedings-mean)**2)
        return consis_loss


    def forward(self,x):
        '''
        :param x: bs,3
        :return:score:(bs,),embedings(T,N,em_dim)
        '''
        embedings = self.cal_node_embeding() # T.N.em_dim
        edge_type = x[:,2]
        src_i = x[:,0]
        tgt_i = x[:,1]
        # 择取特征，
        src_em = embedings[edge_type,src_i].unsqueeze(1) # bs,1,em_dim
        tgt_em = embedings[edge_type,tgt_i].unsqueeze(2) # bs,em_dim,1,等同于（bs,1,em_dim）.transpose(1,2)

        # 择取mk
        mk = self.mk[edge_type] # bs,em_dim,em_dim
        mk = torch.diag_embed(mk) # bs,em_dim,em_dim

        score = src_em @ mk @ tgt_em #bs,1,1
        score = score.squeeze() # (bs,)
        return score,embedings


class GCN(nn.Module):
    def __init__(self,N,adj,K):
        super(GCN, self).__init__()
        self.N=N
        self.K=K
        self.register_buffer('laplacian',cal_laplacian(adj))
    def forward(self,inputs):
        sum = inputs
        for i in range(self.K):
            inputs = torch.sparse.mm(self.laplacian,inputs)
            sum = sum + inputs
        sum = sum/(self.K+1)
        return sum


