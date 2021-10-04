import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

class SingleNegtiveSampler:
    """
    单向采样
    """
    def __init__(self,edges,tgt_nodes):
        '''
        :param neg_num: 采样倍数
        :param edges: 边（edges_num,2）
        :param tgt_nodes: 目标节点集
        '''
        super(SingleNegtiveSampler, self).__init__()
        # num_nodes,other_list
        edges_dir = self.get_edges_dir(edges)
        self.candidate_dir, self.eadge_num_dir = self.get_candidate_dir(edges_dir,set(tgt_nodes))

    @classmethod
    def get_edges_dir(cls,edges):
        '''
        :param edges: 边（edges_num,2）
        :return: 以源节点为起始点的边字典{src_node:[tgt_nodes]}
        '''
        edges_dir={}
        for edge in edges:
            src_node,tgt_node = int(edge[0]),int(edge[1])
            if src_node not in edges_dir:
                edges_dir[src_node] = [tgt_node]
            else:
                edges_dir[src_node].append(tgt_node)
        return edges_dir
    @classmethod
    def get_candidate_dir(cls,edges_dir,tgt_nodes_set):
        '''
        :param edges_dir:
        :param tgt_nodes_set:
        :return:
        '''
        candidate_dir={}
        eadge_num_dir={}
        for node in tqdm(edges_dir):
            tgt_nodes = edges_dir[node]
            candidate_list = list(tgt_nodes_set - set(tgt_nodes))
            candidate_dir[node] = candidate_list
            eadge_num_dir[node] = len(tgt_nodes)
        return candidate_dir,eadge_num_dir

    def sample(self,neg_num):
        '''
        :param neg_num: 采样倍数
        :return: torch.tensor(*,2)
        '''
        neg1=[]
        neg2=[]
        for node in tqdm(self.candidate_dir,leave=False):
            edge_num = self.eadge_num_dir[node]
            neg_list = np.random.choice(self.candidate_dir[node], size=edge_num * neg_num, replace=True)
            neg1.extend([node] * len(neg_list))
            neg2.extend(neg_list)
        result = torch.tensor([neg1,neg2]).long() # 2,*

        return result.transpose(0,1) # *,2

class UDiNegtiveSampler(SingleNegtiveSampler):
    """
    无向图，双向采样
    """

    @classmethod
    def get_edges_dir(cls, edges):
        '''
                :param edges: 边（edges_num,2）
                :return: 以源节点为起始点的边字典{src_node:[tgt_nodes]}
                '''
        edges_dir = {}
        for edge in edges:
            src_node, tgt_node = int(edge[0]), int(edge[1])
            if src_node not in edges_dir:
                edges_dir[src_node] = [tgt_node]
            else:
                edges_dir[src_node].append(tgt_node)
            #     反向
            if tgt_node not in edges_dir:
                edges_dir[tgt_node] = [src_node]
            else:
                edges_dir[tgt_node].append(src_node)
        return edges_dir

    def sample(self, neg_num):
        return super().sample(neg_num//2)


class DiNegtiveSampler:
    """
    有向图，双向采样
    """

    def __init__(self, edges, src_nodes,tgt_nodes):
        self.src2tgt_sampler = SingleNegtiveSampler(edges,tgt_nodes)
        self.tgt2src_sampler = SingleNegtiveSampler(edges[:,[1,0]],src_nodes)

    def sample(self, neg_num):
        '''
        :param neg_num: 采样倍数
        :return: torch.tensor(*,2)
        '''
        r1 = self.src2tgt_sampler.sample(neg_num//2)
        r2 = self.tgt2src_sampler.sample(neg_num//2)
        return torch.cat((r1,r2))
if __name__ == '__main__':
    edge_path = '../data/drug_adj.txt'
    edges = pd.read_csv(edge_path, sep='\t', header=None).to_numpy()
    edges_num = edges.shape[0]
    edges = torch.LongTensor(edges) #n,2

    tgt_node_set = [i for i in range(19122,21257)]
    sampler = UDiNegtiveSampler(edges,tgt_node_set)
    neg = sampler.sample(4)