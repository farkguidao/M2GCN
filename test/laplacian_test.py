from dataloader.dataloader_with_negtive_sampler import NSDataloader
from utils.torch_sparse_utils import *

import numpy as np
import scipy.sparse as sp


if __name__ == '__main__':

    queue = ['drug_drug','drug_protein','protein_protein']
    dataloader = NSDataloader(queue,data_dir='../data')
    adj_list = dataloader.adj_list
    for adj in adj_list:
        la = cal_laplacian(adj).to_dense()
        print(la)
        print('mean:',la.mean())
        print('std:',la.std())
