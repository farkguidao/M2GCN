import torch.sparse as sparse
import torch

def sparse_diags_by_sparse(l):
    n = len(l)
    i = l.indices().expand(2,-1)
    return torch.sparse_coo_tensor(i,l.values(),(n,n),device=l.device)

def sparse_diags(l,device='cpu'):
    n = len(l)
    i = [i for i in range(n)]
    return torch.sparse_coo_tensor([i,i], l, (n, n),device=device)

def sparse_eye(n,device='cpu'):
    i = [i for i in range(n)]
    v = [1.]*n
    return torch.sparse_coo_tensor([i,i],v,(n,n),device=device)
def cal_laplacian(adj):
    n=adj.shape[0]
    A=(adj+sparse_eye(n,device=adj.device)).coalesce()
    degree1 = torch.sparse.sum(A,dim=1)**-0.5
    degree2 = torch.sparse.sum(A,dim=0)**-0.5

    degree1=sparse_diags_by_sparse(degree1)
    degree2=sparse_diags_by_sparse(degree2)

    laplacian = sparse.mm(sparse.mm(degree2,A),degree1)
    return laplacian.coalesce()