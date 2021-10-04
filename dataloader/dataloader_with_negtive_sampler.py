from typing import Optional

import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset,DataLoader

from dataloader.negtive_sampler import *
# 带负采样器的dataloader


class NSDataloader(pl.LightningDataModule):
    def __init__(self,data_queue,batch_size=512,data_dir='./data',num_workers=0,neg_num=4):
        super().__init__()
        self.batch_size = batch_size
        self.neg_num=neg_num
        self.num_workers = num_workers

        self.num_nodes = 21257
        self.pp_num = 19122  # protein种类数
        drug_nodes = [i for i in range(self.pp_num,self.num_nodes)]
        protein_nodes = [i for i in range(self.pp_num)]

        # 规定数据处理队列
        # self.data_queue = ['drug_drug','drug_protein','protein_protein']
        self.data_queue = data_queue

        # data_settings
        self.train_data_setting = {
            'drug_drug':{
                'pos_path': data_dir + '/drug_adj.txt',
                'same_node': True,
                'src_nodes':drug_nodes,
                'tgt_nodes':drug_nodes,
            },
            'drug_protein':{
                'pos_path': data_dir + '/dp_adj.txt',
                'same_node': False,
                'src_nodes': drug_nodes,
                'tgt_nodes': protein_nodes,
            },
            'protein_protein':{
                'pos_path': data_dir + '/protein_adj.txt',
                'same_node': True,
                'src_nodes': protein_nodes,
                'tgt_nodes': protein_nodes,
            }
        }

        self.test_data_setting = {
            'drug_drug':{
                'pos_path':data_dir + '/test_true.txt',
                'neg_path':data_dir + '/test_false.txt'
            }
        }

        self.train_pos_x,self.adj_list,self.sampler_list = self.process_train_data()
        self.test_x,self.test_label = self.process_test_data()
        # reverse test_x
        # self.test_x = self.test_x[:,[1,0,2]]


    def process_train_data(self):
        """
        读取edge,创建adj矩阵列表,创建采样器,把数据封装好
        :return: train_pos_x:tensor,adj_list:list,sampler_list:list
        """
        adj_list=[]
        sampler_list=[]
        train_pos_x=[]
        for i, key in enumerate(self.data_queue):
            if key not in self.train_data_setting:
                continue
            setting = self.train_data_setting[key]
            # 读取edges
            edges = self.read_edges(setting['pos_path']) # *,2
            # 生成adj
            adj = self.create_adj(self.num_nodes,edges)
            # 创建sampler
            if setting['same_node']:
                sampler = UDiNegtiveSampler(edges,setting['tgt_nodes'])
            else:
                sampler = DiNegtiveSampler(edges,setting['src_nodes'],setting['tgt_nodes'])
            #  添加边的类型
            edges = self.add_edge_type(edges,i)
            train_pos_x.append(edges)
            adj_list.append(adj)
            sampler_list.append(sampler)
        train_pos_x = torch.cat(train_pos_x)
        return train_pos_x,adj_list,sampler_list

    def process_test_data(self):
        """
        :return:test_x,test_label
        """
        test_x,test_label=[],[]
        for i,key in enumerate(self.data_queue):
            if key not in self.test_data_setting:
                continue
            setting = self.test_data_setting[key]
            pos_edges = self.read_edges(setting['pos_path'])
            neg_edges = self.read_edges(setting['neg_path'])

            pos_num,neg_num = pos_edges.shape[0],neg_edges.shape[0]

            edges = self.add_edge_type(torch.cat([pos_edges,neg_edges]),i)
            test_x.append(edges)

            # label
            label=torch.zeros(pos_num+neg_num)
            label[:pos_num]=1.
            test_label.append(label)
        test_x = torch.cat(test_x)
        test_label = torch.cat(test_label)
        return test_x,test_label

    def add_edge_type(self,edges,edge_type):
        # 为边加上类型
        t = torch.LongTensor(edges.shape[0], 1)
        t[:] = edge_type
        edges = torch.cat([edges, t], dim=1)  # *,3
        return edges
    def sample(self,neg_num):
        '''
        :param neg_num:采样倍数
        :return:
        '''
        train_neg_x=[]
        for i,sampler in enumerate(self.sampler_list):
            neg_edges=sampler.sample(neg_num)
            neg_edges=self.add_edge_type(neg_edges,i)
            train_neg_x.append(neg_edges)
        train_neg_x=torch.cat(train_neg_x)
        return train_neg_x

    @classmethod
    def create_adj(cls,num_nodes,edges):
        '''
        :param num_nodes:总的节点数量
        :param edges: tensor(*,2)
        :return:
        '''
        value = torch.ones(edges.shape[0])
        # 创建稀疏图
        adj = torch.sparse_coo_tensor(edges.transpose(0,1),value,(num_nodes,num_nodes))
        # 无向图的对称边
        adj=(adj+adj.t()).coalesce()
        return adj

    @classmethod
    def read_edges(cls,edge_path):
        '''
        :param edge_path:
        :return: tensor(*,2)
        '''
        # 读取边
        data = pd.read_csv(edge_path, sep='\t', header=None)
        edges = data.drop_duplicates().to_numpy() #去除重复行
        edges = torch.LongTensor(edges)
        return edges


    def make_meal(self):
        train_neg_x = self.sample(self.neg_num)
        train_x = torch.cat([self.train_pos_x,train_neg_x])
        train_label = torch.zeros(train_x.shape[0])
        train_label[:self.train_pos_x.shape[0]]=1.
        return train_x,train_label


    def train_dataloader(self):
        # self.neg_num = 10 - self.neg_num
        train_x, train_label = self.make_meal()
        dataset = TensorDataset(train_x, train_label)
        return DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = TensorDataset(self.test_x, self.test_label)
        return DataLoader(dataset=dataset, batch_size=self.test_x.shape[0])

    def test_dataloader(self):
        return self.val_dataloader()

if __name__ == '__main__':
    dataloader = NSDataloader(batch_size=512*8,data_dir='../data')
    for i in range(10):
        data = dataloader.sample(4)
        print(i)
        if(data.isnan().sum()!=0):
            print('NAN!')