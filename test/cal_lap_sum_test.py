from utils.torch_sparse_utils import *
def cal_lap_sum(adj,k):
    la = cal_laplacian(adj)
    temp = sparse_eye(adj.shape[0])
    sum = temp
    # sum = E+L+L^2+...+L^K
    for i in range(k):
        temp = torch.sparse.mm(la, temp)
        sum = (sum + temp).coalesce()
    sum = sum / (k + 1)
    return sum

from dataloader.dataloader_with_negtive_sampler import NSDataloader
if __name__ == '__main__':

    dataloader = NSDataloader(batch_size=512*8,data_dir='../data')
    adj_list = dataloader.adj_list
    sum=cal_lap_sum(adj_list[1],8)